# environment.py

import random
import json
import os
import logging
import concurrent.futures
import statistics
from datetime import datetime
from core import parameters

logger = logging.getLogger(__name__)
from core.agent import Agent
from core.institution import SanctioningInstitution, SanctionFreeInstitution
from modules.tom_module import TomModule
from modules.democracy_module import DemocracyModule
from modules.oracle import Oracle
from core.subsidy import SubsidyModule
from core.loss_damage_fund import LossDamageFund
from core.scenario_config import get_scenario_config
if parameters.GOSSIP_ENABLED:
    from modules.gossip_module import GossipModule

class Environment:
    def __init__(self, agents):
        self.agents = agents
        self.current_round = 0
        self.history = []
        self.si = SanctioningInstitution()
        self.sfi = SanctionFreeInstitution()
        self.results = []
        self.constitutional_history = []  # Phase 2: log of democracy outcomes
        self.current_shock = {'occurred': False, 'severity': 0.0, 'damages': {}}
        self.last_ldf_summary = {
            'pool_start': 0.0,
            'contributions_total': 0.0,
            'payouts_total': 0.0,
            'pool_end': 0.0,
        }
        random.seed(parameters.SEED)

        # Phase 2: shared module instances (use first agent's API client)
        api_client = agents[0].api_client if agents else None
        self.tom_module = TomModule(api_client) if (api_client and parameters.TOM_ENABLED) else None
        self.democracy_module = DemocracyModule(api_client) if (api_client and parameters.DEMOCRACY_ENABLED) else None
        self.subsidy_module = SubsidyModule() if parameters.SUBSIDY_ENABLED else None
        self.ldf_module = LossDamageFund() if parameters.LDF_ENABLED else None
        self.gossip_module = GossipModule() if parameters.GOSSIP_ENABLED else None

    def run_simulation(self):
        """
        Runs the simulation for the specified number of rounds.
        """
        for round_number in range(1, parameters.NUM_ROUNDS + 1):
            self.current_round = round_number
            progress_str = f" [Run {getattr(parameters, 'CURRENT_RUN', '?')}/{getattr(parameters, 'TOTAL_RUNS', '?')}]" if hasattr(parameters, 'CURRENT_RUN') else ""
            logger.info(f"{progress_str} Starting Round {self.current_round}/{parameters.NUM_ROUNDS}")
            self.run_round()

            # Phase 2: Theory of Mind audit after every round
            if self.tom_module:
                self.run_tom_audit(round_number)

            # Phase 2: Democratic vote every DEMOCRACY_INTERVAL rounds
            if self.democracy_module and round_number % parameters.DEMOCRACY_INTERVAL == 0:
                # Phase 3: Build oracle with current round history before session
                oracle = Oracle(round_history=self.results, agents=self.agents)
                result = self.democracy_module.run_constitutional_session(
                    self.agents, round_number, oracle=oracle
                )
                self.constitutional_history.append({
                    'round': round_number,
                    **result
                })
                # Attach democracy result to the latest round data
                if self.results:
                    self.results[-1]['constitutional_change'] = result


    def run_tom_audit(self, round_number):
        """
        Phase 2: Run Theory of Mind audits for every agent.
        Each agent scores every other agent's behavioural consistency.
        The peer-average scores update each agent's reputation.
        Phase 2b: Gossip distributed to agents.
        """
        # Tally incoming trust scores for each agent: {agent_id: [scores...]}
        incoming_scores: dict = {a.agent_id: [] for a in self.agents}
        all_audits_this_round = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(self.agents), max(1, int(parameters.LLM_MAX_CONCURRENCY)))
        ) as executor:
            futures = {executor.submit(self.tom_module.audit_round, evaluator, self.agents, round_number): evaluator for evaluator in self.agents}
            for future in concurrent.futures.as_completed(futures):
                evaluator = futures[future]
                scores = future.result()
                for target_id, data in scores.items():
                    score = data['score']
                    reasoning = data['reasoning']
                    evaluator.tom_scores[target_id] = score
                    incoming_scores[target_id].append(score)
                    if self.gossip_module:
                        all_audits_this_round.append({
                            'source': evaluator.agent_id,
                            'target': target_id,
                            'score': score,
                            'reasoning': reasoning
                        })

        # Update each agent's reputation as the average of all incoming scores
        for agent in self.agents:
            scores_received = incoming_scores.get(agent.agent_id, [])
            if scores_received:
                agent.reputation = sum(scores_received) / len(scores_received)
            # Reputation stays at the previous value if no scores received

        # Phase 2b: Compile and distribute gossip if enabled
        if self.gossip_module:
            self.gossip_module.compile_gossip(all_audits_this_round)
            for agent in self.agents:
                agent.recent_gossip = self.gossip_module.get_gossip_for_agent(agent)

        rep_str = ", ".join(
            f"Agent {a.agent_id}: {a.reputation:.1f}"
            for a in self.agents
        )
        logger.info(f"[ToM] Reputations after Round {round_number}: [{rep_str}]")
        
        if self.gossip_module and self.gossip_module.gossip_bulletin:
            logger.info(f"{'='*20} SOCIAL GOSSIP BULLETIN {'='*20}")
            for gossip in self.gossip_module.gossip_bulletin:
                logger.info(f"Agent {gossip['source']} on Agent {gossip['target']}: \"{gossip['reasoning'][:150]}...\"")
            logger.info(f"{'='*64}")

    def run_round(self):
        """
        Executes a single round of the simulation.
        """
        # Reset institutions for the new round
        self.si.reset_institution(round_number=self.current_round)
        self.sfi.reset_institution(round_number=self.current_round)

        # Agents choose institutions
        def setup_agent(agent):
            logger.debug(f"Agent {agent} started")
            agent.reset_for_new_round()
            scenario_name = str(getattr(parameters, 'SCENARIO', '')).lower()
            if scenario_name == 'climate':
                scenario_name = 'ldf'

            climate_mode = (
                scenario_name == 'ldf'
                or bool(getattr(parameters, 'CLIMATE_SHOCK_ENABLED', False))
                or bool(getattr(parameters, 'LDF_ENABLED', False))
            )

            if climate_mode:
                if getattr(agent, 'agent_group', 'developing') == 'developed':
                    agent.institution_choice = 'SI'
                    agent.institution_reasoning = 'Climate/LDF mode defaults developed countries to the binding treaty.'
                    agent.institution_facts_used = ['Developed countries are routed to SI in climate/LDF mode.']
                    agent.institution_deepseek_think = ''
                    agent.log_debug(self.current_round, "stage_0_institution", "Climate/LDF mode defaults developed countries to SI.", '{"institution_choice": "SI"}')
                else:
                    agent.institution_choice = 'SFI'
                    agent.institution_reasoning = 'Climate/LDF mode defaults developing countries to the non-binding agreement.'
                    agent.institution_facts_used = ['Developing countries are routed to SFI in climate/LDF mode.']
                    agent.institution_deepseek_think = ''
                    agent.log_debug(self.current_round, "stage_0_institution", "Climate/LDF mode defaults developing countries to SFI.", '{"institution_choice": "SFI"}')
            else:
                agent.choose_institution(self.current_round)
            return agent

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(self.agents), max(1, int(parameters.LLM_MAX_CONCURRENCY)))
        ) as executor:
            futures = [executor.submit(setup_agent, agent) for agent in self.agents]
            for future in concurrent.futures.as_completed(futures):
                agent = future.result()
                # Add agent to the chosen institution
                if agent.institution_choice == 'SI':
                    self.si.add_member(agent)
                    agent.current_group = self.si
                else:
                    self.sfi.add_member(agent)
                    agent.current_group = self.sfi
                
                logger.info(f"Agent {agent.agent_id} chose {agent.institution_choice}. Reasoning: \"{agent.institution_reasoning[:150]}...\"")

        # Collect contributions in each institution
        self.si.collect_contributions()
        self.sfi.collect_contributions()

        # Distribute public goods in each institution
        self.si.distribute_public_goods()
        self.sfi.distribute_public_goods()

        # Handle punishments and rewards in SI
        self.si.handle_punishments_and_rewards()
        self.si.apply_punishments_and_rewards()

        # Phase 4: Subsidy redistribution
        if self.subsidy_module:
            round_contributions = {a.agent_id: a.contribution for a in self.agents}
            subsidies = self.subsidy_module.compute_subsidies(self.agents, round_contributions)
            for agent_id, bonus in subsidies.items():
                agent = next(a for a in self.agents if a.agent_id == agent_id)
                agent.last_subsidy = bonus
                currency = get_scenario_config(parameters.SCENARIO)['currency_name']
                logger.info(f"Agent {agent_id} received subsidy: +{bonus} {currency}")

        # Climate shocks and Loss & Damage Fund (optional; disabled by default).
        self._apply_climate_shock_and_ldf()
                    
        # Calculate payoffs and update agents
        self.calculate_payoffs()

        # Update history and agents' personal histories
        self.record_round_history()

    def _apply_climate_shock_and_ldf(self):
        # Defaults when modules are disabled or no event occurs.
        self.current_shock = {'occurred': False, 'severity': 0.0, 'damages': {}}
        self.last_ldf_summary = {
            'pool_start': self.ldf_module.pool_balance if self.ldf_module else 0.0,
            'contributions_collected': False,
            'contributions_total': 0.0,
            'payouts_total': 0.0,
            'pool_end': self.ldf_module.pool_balance if self.ldf_module else 0.0,
        }

        if self.ldf_module:
            pool_start = self.ldf_module.pool_balance
            should_collect = self.ldf_module.should_collect_contributions(self.current_round)
            contributions_total = 0.0

            if should_collect:
                contributions = self.ldf_module.collect_contributions(self.agents)
                contributions_total = sum(contributions.values())

            self.last_ldf_summary.update({
                'pool_start': pool_start,
                'contributions_collected': should_collect,
                'contributions_total': contributions_total,
                'pool_end': self.ldf_module.pool_balance,
            })

        if not parameters.CLIMATE_SHOCK_ENABLED:
            return

        # Deterministic schedule overrides stochastic roll when enabled.
        if getattr(parameters, 'CLIMATE_SHOCK_DETERMINISTIC', False):
            # Find a scheduled shock for the current round (exact match)
            severity = None
            for r, s in getattr(parameters, 'CLIMATE_SHOCK_SCHEDULE', []):
                if r == self.current_round:
                    severity = s
                    break

            if severity is None:
                # No scheduled shock this round
                return

        else:
            # Original stochastic logic
            shock_roll = random.random()
            if shock_roll >= parameters.CLIMATE_SHOCK_BASE_PROB:
                return
            severity = random.uniform(parameters.CLIMATE_SHOCK_SEVERITY_MIN, parameters.CLIMATE_SHOCK_SEVERITY_MAX)

        damages = {}

        for agent in self.agents:
            damage = parameters.CLIMATE_DAMAGE_BASE * severity * max(0.0, agent.vulnerability)
            damages[agent.agent_id] = damage
            agent.climate_damage_taken_round = damage
            agent.climate_damage_taken_cumulative += damage

        payouts = {}
        if self.ldf_module:
            payouts = self.ldf_module.distribute_payouts(self.agents, damages)
            self.last_ldf_summary.update({
                'payouts_total': sum(payouts.values()),
                'pool_end': self.ldf_module.pool_balance,
            })

        self.current_shock = {
            'occurred': True,
            'severity': severity,
            'damages': damages,
            'payouts': payouts,
        }

        if self.ldf_module:
            self.ldf_module.append_round_log({
                'round_number': self.current_round,
                'shock_occurred': self.current_shock.get('occurred', False),
                'shock_severity': self.current_shock.get('severity', 0.0),
                'damages': self.current_shock.get('damages', {}),
                'payouts': self.current_shock.get('payouts', {}),
                **self.last_ldf_summary,
            })

    def calculate_payoffs(self):
        """
        Calculates the total payoffs for all agents after considering contributions,
        punishments, rewards, and updates their cumulative payoffs.
        """
        # Update payoffs for all agents
        for agent in self.agents:
            # Determine institution-specific stage 1 payoff
            if agent.institution_choice == 'SI':
                stage1_payoff = self.si.stage1_payoffs.get(agent.agent_id, 0)
                # Apply stage 1 results to wealth before calculating stage 2
                agent.wealth += stage1_payoff
                stage2_payoff = agent.get_stage2_payoff()
            else:
                stage1_payoff = self.sfi.stage1_payoffs.get(agent.agent_id, 0)
                # Apply stage 1 results to wealth
                agent.wealth += stage1_payoff
                stage2_payoff = 0
            
            subsidy = getattr(agent, 'last_subsidy', 0)
            ldf_transfer = self.current_shock.get('payouts', {}).get(agent.agent_id, 0.0)
            climate_damage = self.current_shock.get('damages', {}).get(agent.agent_id, 0.0)
            agent.ldf_payout_round = ldf_transfer
            agent.net_climate_transfer_round = ldf_transfer - climate_damage
            
            total_round_payoff = stage1_payoff + stage2_payoff + subsidy + ldf_transfer - climate_damage
            agent.update_payoff(total_round_payoff)
            agent.wealth += (stage2_payoff + subsidy + ldf_transfer - climate_damage)
            agent.wealth = max(0.0, agent.wealth)

            payoff_details = f"S1: {stage1_payoff:.1f}, S2: {stage2_payoff:.1f}, LDF: {ldf_transfer:.1f}, Damage: {climate_damage:.1f}"
            logger.info(f"Agent {agent.agent_id} in {agent.institution_choice} (Payoff: {agent.round_payoff:.2f} | {payoff_details})")
            logger.debug(f"  - LLM Reasoning: {agent.contribution_reasoning[:120]}...")

    def _compute_gini(self, values):
        """Gini coefficient for non-negative values."""
        if not values:
            return 0.0
        vals = [max(0.0, float(v)) for v in values]
        n = len(vals)
        mean_v = statistics.mean(vals)
        if n == 0 or mean_v == 0:
            return 0.0
        sorted_vals = sorted(vals)
        cumulative = 0.0
        for i, x in enumerate(sorted_vals, start=1):
            cumulative += i * x
        return (2 * cumulative) / (n * sum(sorted_vals)) - (n + 1) / n

    def record_round_history(self):
        """
        Records the outcome of the round for analysis and provides feedback to agents.
        """
        # Collect data for the round
        round_data = {
            'round_number': self.current_round,
            'si_members': [agent.agent_id for agent in self.si.members],
            'sfi_members': [agent.agent_id for agent in self.sfi.members],
            'si_total_contribution': self.si.total_contribution,
            'sfi_total_contribution': self.sfi.total_contribution,
            'si_avg_contribution': self.si.get_average_contribution(),
            'sfi_avg_contribution': self.sfi.get_average_contribution(),
            'shock_occurred': self.current_shock.get('occurred', False),
            'shock_severity': self.current_shock.get('severity', 0.0),
            'gross_damage_total': sum(self.current_shock.get('damages', {}).values()),
            'net_damage_total': sum(
                max(0.0, self.current_shock.get('damages', {}).get(a.agent_id, 0.0) - getattr(a, 'ldf_payout_round', 0.0))
                for a in self.agents
            ),
            'ldf_pool_start': self.last_ldf_summary.get('pool_start', 0.0),
            'ldf_contributions_total': self.last_ldf_summary.get('contributions_total', 0.0),
            'ldf_payouts_total': self.last_ldf_summary.get('payouts_total', 0.0),
            'ldf_pool_end': self.last_ldf_summary.get('pool_end', 0.0),
            'agents': {}
        }

        ratios = []
        for a in self.agents:
            cap = a.get_stage1_contribution_cap() if hasattr(a, 'get_stage1_contribution_cap') else parameters.ENDOWMENT_STAGE_1
            ratios.append(a.contribution / cap if cap > 0 else 0.0)
        round_data['cooperation_rate'] = statistics.mean(ratios) if ratios else 0.0
        round_data['gini_wealth'] = self._compute_gini([a.wealth for a in self.agents])

        # Calculate cumulative payoffs for ranking
        agent_cumulative_payoffs = {
            agent.agent_id: agent.cumulative_payoff for agent in self.agents
        }
        # Calculate ranks based on cumulative payoff
        sorted_payoffs = sorted(
            agent_cumulative_payoffs.items(), key=lambda x: x[1], reverse=True
        )
        agent_ranks = {
            agent_id: rank + 1 for rank, (agent_id, _) in enumerate(sorted_payoffs)
        }

        total_agents = len(self.agents)

        # Record detailed agent data and provide feedback
        for agent in self.agents:
            # Retrieve stage payoffs
            if agent.institution_choice == 'SI':
                stage1_payoff = self.si.stage1_payoffs.get(agent.agent_id, 0)
                stage2_payoff = agent.get_stage2_payoff()
            else:
                stage1_payoff = self.sfi.stage1_payoffs.get(agent.agent_id, 0)
                stage2_payoff = 0

            # Total round payoff — use the already-assembled round_payoff so
            # subsidy is included and this matches what cumulative_payoff tracks.
            total_round_payoff = agent.round_payoff

            # Retrieve rank
            rank = agent_ranks[agent.agent_id]

            # Prepare agent data with detailed assigned punishments and rewards
            agent_data = {
                'institution_choice': agent.institution_choice,
                'institution_reasoning': agent.institution_reasoning,
                'institution_facts_used': getattr(agent, 'institution_facts_used', []),
                'institution_deepseek_think': getattr(agent, 'institution_deepseek_think', ''),
                'institution_parser_meta': getattr(agent, 'institution_parser_meta', {}),
                'contribution': agent.contribution,
                'contribution_reasoning': agent.contribution_reasoning,
                'contribution_facts_used': getattr(agent, 'contribution_facts_used', []),
                'contribution_deepseek_think': getattr(agent, 'contribution_deepseek_think', ''),
                'contribution_parser_meta': getattr(agent, 'contribution_parser_meta', {}),
                'stage1_payoff': stage1_payoff,
                'stage2_payoff': stage2_payoff,
                'payoff': total_round_payoff,
                'cumulative_payoff': agent.cumulative_payoff,
                'strategy': agent.strategy,
                'agent_group': getattr(agent, 'agent_group', 'developing'),
                'wealth': getattr(agent, 'wealth', agent.cumulative_payoff),
                'vulnerability': getattr(agent, 'vulnerability', 1.0),
                'historical_emissions': getattr(agent, 'historical_emissions', 0.0),
                'contribution_capacity': getattr(agent, 'contribution_capacity', 1.0),
                'received_punishments': agent.received_punishments,
                'received_rewards': agent.received_rewards,
                'assigned_punishments': agent.assigned_punishments,
                'assigned_rewards': agent.assigned_rewards,
                'punishment_reasoning': agent.punishment_reasoning,
                'deanonymized_punishment_reasoning': agent.deanonymized_punishment_reasoning,
                'punishment_facts_used': getattr(agent, 'punishment_facts_used', []),
                'punishment_justifications': getattr(agent, 'punishment_justifications', {}),
                'punishment_deepseek_think': getattr(agent, 'punishment_deepseek_think', ''),
                'punishment_parser_meta': getattr(agent, 'punishment_parser_meta', {}),
                'reputation': round(agent.reputation, 2),            # Phase 2
                'tom_scores': {str(k): round(v, 2) for k, v in agent.tom_scores.items()},  # Phase 2
                'rank': f"{rank} out of {total_agents}",
                'subsidy': getattr(agent, 'last_subsidy', 0), # Phase 4
                'climate_damage_taken_round': getattr(agent, 'climate_damage_taken_round', 0.0),
                'climate_damage_taken_cumulative': getattr(agent, 'climate_damage_taken_cumulative', 0.0),
                'ldf_contribution_round': getattr(agent, 'ldf_contribution_round', 0.0),
                'ldf_payout_round': getattr(agent, 'ldf_payout_round', 0.0),
                'net_climate_transfer_round': getattr(agent, 'net_climate_transfer_round', 0.0),
                'parsing_failures': getattr(agent, 'parsing_failures', 0),
                'rule_of_law_blocks': getattr(agent, 'rule_of_law_blocks', 0),
                'belief_state': getattr(agent, 'belief_state', {}),
            }

            # Add the agent data to the round data
            round_data['agents'][agent.agent_id] = agent_data

            # Prepare feedback for the agent
            avg_contribution_institution = (
                self.si.get_average_contribution()
                if agent.institution_choice == 'SI'
                else self.sfi.get_average_contribution()
            )

            feedback = agent_data.copy()
            feedback['round_number'] = self.current_round
            feedback['avg_payoff_SI'] = (
                sum([a.round_payoff for a in self.si.members]) / len(self.si.members)
                if self.si.members
                else 0
            )
            feedback['avg_payoff_SFI'] = (
                sum([a.round_payoff for a in self.sfi.members]) / len(self.sfi.members)
                if self.sfi.members
                else 0
            )
            feedback['avg_contribution_institution'] = avg_contribution_institution

            # Update agent's personal history
            agent.update_history(feedback)

        # Build anonymous data for each agent
        for agent in self.agents:
            anonymous_data_list = []
            for other_agent in self.agents:
                if other_agent.agent_id != agent.agent_id:
                    # Get other agent's data from round_data
                    other_agent_data = round_data['agents'][other_agent.agent_id]
                    # Extract the data needed
                    anonymous_entry = {
                        'actual_agent_id': other_agent.agent_id,
                        'institution_choice': other_agent_data['institution_choice'],
                        'agent_group': other_agent_data.get('agent_group', 'unknown'),
                        'wealth': other_agent_data.get('wealth', 0.0),
                        'contribution': other_agent_data['contribution'],
                        'received_punishments': other_agent_data['received_punishments'],
                        'received_rewards': other_agent_data['received_rewards'],
                        'stage1_payoff': other_agent_data['stage1_payoff'],
                        'stage2_payoff': other_agent_data['stage2_payoff'],
                        'total_round_payoff': other_agent_data['payoff']
                    }
                    # Note: We do not include the detailed assigned punishments and rewards in the anonymous data
                    # to maintain anonymity and prevent deanonymization.
                    anonymous_data_list.append(anonymous_entry)
            # Set the agent's current_round_anonymous_data
            agent.current_round_anonymous_data = anonymous_data_list

        # --- Belief Tracking: parallel update of each agent's belief state ---
        if getattr(parameters, 'BELIEF_TRACKING_ENABLED', False):
            feedback_map = {}
            for agent in self.agents:
                agent_feedback = round_data['agents'].get(agent.agent_id, {})
                agent_feedback['round_number'] = self.current_round
                feedback_map[agent.agent_id] = agent_feedback

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(self.agents), max(1, int(parameters.LLM_MAX_CONCURRENCY)))
            ) as executor:
                futures = {
                    executor.submit(
                        agent.update_beliefs,
                        feedback_map[agent.agent_id],
                        agent.current_round_anonymous_data
                    ): agent
                    for agent in self.agents
                }
                for future in concurrent.futures.as_completed(futures):
                    agent = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.warning(f"[Belief Update] Agent {agent.agent_id} failed: {e}")

            # Snapshot the updated belief states into round_data for results JSON
            for agent in self.agents:
                round_data['agents'][agent.agent_id]['belief_state'] = getattr(agent, 'belief_state', {})

        # Append round data to environment history
        self.history.append(round_data)

        # Optionally, save data for analysis
        self.results.append(round_data)

    def save_results(self, model_name, num_agents, num_rounds):
        """
        Saves the simulation results to a timestamped JSON file inside a 'results/' subfolder.
        Filename format: results/simulation_<model>_<N>agents_<R>rounds_<YYYYMMDD_HHMMSS>.json
        """
        # Build the results directory next to this script
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_component = model_name.replace('/', '_').replace(':', '_')
        batch_name = getattr(parameters, 'BATCH_NAME', 'run')
        seed_val = getattr(parameters, 'SEED', 'noseed')
        filename = (
            f"simulation_{model_component}_{batch_name}_seed{seed_val}_{num_agents}agents"
            f"_{num_rounds}rounds_{timestamp}.json"
        )
        filepath = os.path.join(results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)

        logger.info(f"Simulation results saved to '{filepath}'.")