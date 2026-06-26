# agent.py

"""
This module defines the Agent class representing participants in the experiment.

Agents use Large Language Models (LLMs) via API calls to make their decisions.

Agents will:

- Choose institutions based on prompts sent to the LLM.

- Decide on contributions by generating prompts and parsing the LLM's responses.

- Assign punishments or rewards in the Sanctioning Institution (SI) based on LLM output.

The agent's internal monologue or reasoning is captured and stored under attributes:

- 'institution_reasoning' for institution choice

- 'contribution_reasoning' for contribution decision

- 'punishment_reasoning' for punishment and reward assignments

Dependencies:

- Requires OllamaClient for local LLM interaction.
"""

import logging
import random
from core import parameters
import os
import json
from core.utils import robust_json_loads, uses_climate_budget

logger = logging.getLogger(__name__)

from prompts.prompt_generator import (
    construct_institution_choice_prompt,
    construct_contribution_prompt,
    construct_punishment_prompt,
    get_past_actions_string
)

from parsing import (
    parse_institution_choice_response,
    parse_contribution_response_v2,
    parse_punishment_response,
    deanonymize_reasoning
)


def _schema_repair_prompt(base_prompt, stage_name):
    return (
        f"{base_prompt}\n\n"
        f"IMPORTANT RETRY ({stage_name}): Your previous response did not strictly satisfy the JSON contract. "
        "Return ONLY one valid JSON object that exactly follows the required schema and key names. "
        "No markdown, no code fences, no extra text."
    )

class Agent:
    def __init__(self, agent_id, api_client):
        """
        Initialize an Agent.

        Parameters:
        - agent_id (int): Unique identifier for the agent.
        - api_client: The OllamaClient used for all LLM calls.
        """
        self.agent_id = agent_id
        self.api_client = api_client
        self.summary_api_client = api_client  # Alias for deanonymization calls (same client)
        self.initial_wealth = float(parameters.INITIAL_TOKENS)

        self.institution_choice = None  # 'SI' or 'SFI'
        self.contribution = 0
        self.cumulative_payoff = parameters.INITIAL_TOKENS  # Total accumulated payoff
        self.round_payoff = 0  # Payoff for the current round
        self.history = []
        self.current_group = None  # Reference to the institution/group the agent is currently in

        # Round-specific attributes
        self.received_punishments = 0
        self.received_rewards = 0
        self.assigned_punishments = {}  # Dict of agent_id: tokens assigned
        self.assigned_rewards = {}  # Dict of agent_id: tokens assigned

        # Additional attributes for LLM interaction
        self.round_number = 0  # Current round number

        # Add 'strategy' attribute for compatibility
        # Identify agent type for logging/analysis
        self.strategy = 'LLM'
        self.llm_persona = 'DEFAULT'

        # --- Phase 5: Heterogeneous climate profile ---
        self.agent_group = 'developing'
        self.wealth = float(parameters.INITIAL_TOKENS)
        self.vulnerability = float(parameters.DEVELOPING_VULNERABILITY)
        self.historical_emissions = float(parameters.DEVELOPING_HISTORICAL_EMISSIONS)
        self.contribution_capacity = float(parameters.DEVELOPING_CONTRIBUTION_CAPACITY)

        # Climate shock + LDF accounting
        self.climate_damage_taken_round = 0.0
        self.climate_damage_taken_cumulative = 0.0
        self.ldf_contribution_round = 0.0
        self.ldf_payout_round = 0.0
        self.net_climate_transfer_round = 0.0

        # Attributes to store reasoning
        self.institution_reasoning = ''
        self.contribution_reasoning = ''
        self.punishment_reasoning = ''
        self.institution_deepseek_think = ''
        self.contribution_deepseek_think = ''
        self.punishment_deepseek_think = ''
        self.institution_facts_used = []
        self.contribution_facts_used = []
        self.punishment_facts_used = []

        # Attribute to store anonymous data history
        self.anonymous_data_history = []  # List of dicts storing data for previous rounds
        self.current_round_anonymous_data = None  # Data collected in the current round

        # Add attribute to store mapping of anonymized IDs to actual agent IDs
        self.anonymized_id_mapping = {}  # Mapping of anonymized agent numbers to actual agent IDs for the current prompt

        # For deanonymized reasoning
        self.deanonymized_punishment_reasoning = ''  # Deanonymized version of punishment reasoning

        # Stable pseudonym mapping to prevent cross-round identity confusion
        self.pseudonym_mapping = {}  # {actual_agent_id: stable_pseudonym_integer}
        self.reverse_pseudonym_mapping = {} # {stable_pseudonym_integer: actual_agent_id}

        # --- Phase 4: Subsidy & Curiosity ---
        self.last_subsidy = 0       # tokens received in current round
        self.explored_params = set() # For curiosity module: unique parameters proposed
        self.history_institutions = [] # Track all past choices
        self.history_contributions = [] # Track last few contributions

        # --- Reliability tracking ---
        self.parsing_failures = 0
        self.rule_of_law_blocks = 0
        self.institution_parser_meta = {}
        self.contribution_parser_meta = {}
        self.punishment_parser_meta = {}

        # --- Phase 2: Theory of Mind & Reputation ---
        self.tom_scores = {}        # {other_agent_id: trust_score (1-10)} — updated each round
        self.reputation = 5.0       # Peer-average trust score (default = neutral)
        self.stated_intent = ''     # Saved contribution reasoning before the action (for ToM audit)
        self.tom_audit_log = []     # Log of all ToM audit entries this agent has made
        self.recent_gossip = ""     # Phase 2b: Gossip bulletin from the previous round

        # --- Belief Tracking (Working Memory / Scratchpad) ---
        self.belief_state = {
            "trust_levels": {},
            "institutional_strategy": "No prior experience — exploring options.",
            "observations": "No rounds played yet."
        }


        # Initialize pseudonyms for this agent to prevent the anonymization null-routing bug
        self._ensure_pseudonyms_initialized()

    def _uses_climate_budget(self):
        return uses_climate_budget()

    def get_stage1_contribution_cap(self):
        if self._uses_climate_budget():
            return max(parameters.MIN_CONTRIBUTION, int(self.wealth))
        return parameters.ENDOWMENT_STAGE_1

    def choose_institution(self, round_number):
        """
        Decide whether to join the Sanctioning Institution (SI) or the Sanction-Free Institution (SFI)
        by generating a prompt and sending it to the LLM.
        """
        self.round_number = round_number

        prompt = construct_institution_choice_prompt(self, round_number)
        temperature, top_p = self._persona_sampling_profile("institution")

        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt,
            response_format={"type": "json_object"},
            max_tokens=768,
            temperature=temperature,
            top_p=top_p
        )

        choice, reasoning, facts_used, deepseek_think, parser_meta = parse_institution_choice_response(response, self.agent_id)
        if parser_meta.get('fallback_used', False):
            repair_response = self.api_client.send_request(
                model_name=self.api_client.deployment_name,
                prompt=_schema_repair_prompt(prompt, "Institution Choice"),
                response_format={"type": "json_object"},
                max_tokens=768,
                temperature=temperature,
                top_p=top_p
            )
            retry_choice, retry_reasoning, retry_facts, retry_deepseek, retry_meta = parse_institution_choice_response(repair_response, self.agent_id)
            if not retry_meta.get('fallback_used', False):
                response = repair_response
                choice, reasoning, facts_used, deepseek_think, parser_meta = retry_choice, retry_reasoning, retry_facts, retry_deepseek, retry_meta

        if parser_meta.get('fallback_used', False):
            self.parsing_failures += 1
        
        self.institution_choice = choice
        self.institution_reasoning = reasoning
        self.institution_facts_used = facts_used
        self.institution_deepseek_think = deepseek_think
        self.institution_parser_meta = parser_meta
        
        self.log_debug(round_number, "stage_0_institution", prompt, response)

    def _ensure_pseudonyms_initialized(self):
        """
        Creates a stable, randomized mapping of actual agent IDs to pseudonyms (1..N).
        This ensures that 'Agent X' always refers to the same individual across all rounds
        for this specific observer, preventing history vs. current round confusion.
        """
        if not self.pseudonym_mapping:
            other_agents = [a for a in range(parameters.NUM_AGENTS) if a != self.agent_id]
            rng = random.Random(parameters.SEED + self.agent_id)
            rng.shuffle(other_agents)
            
            for i, actual_id in enumerate(other_agents):
                pseudonym = i + 1
                self.pseudonym_mapping[actual_id] = pseudonym
                self.reverse_pseudonym_mapping[pseudonym] = actual_id



    def decide_contribution(self, group_state):
        """
        Decide how much to contribute to the public good using the LLM.
        """
        prompt = construct_contribution_prompt(self, group_state)
        temperature, top_p = self._persona_sampling_profile("contribution")
        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt,
            response_format={"type": "json_object"},
            max_tokens=768,
            temperature=temperature,
            top_p=top_p
        )
        contribution, llm_reasoning, facts_used, deepseek_think, parser_meta = parse_contribution_response_v2(response, self)
        if parser_meta.get('fallback_used', False):
            repair_response = self.api_client.send_request(
                model_name=self.api_client.deployment_name,
                prompt=_schema_repair_prompt(prompt, "Contribution Choice"),
                response_format={"type": "json_object"},
                max_tokens=768,
                temperature=temperature,
                top_p=top_p
            )
            retry_contrib, retry_reasoning, retry_facts, retry_deepseek, retry_meta = parse_contribution_response_v2(repair_response, self)
            if not retry_meta.get('fallback_used', False):
                response = repair_response
                contribution, llm_reasoning, facts_used, deepseek_think, parser_meta = retry_contrib, retry_reasoning, retry_facts, retry_deepseek, retry_meta

        if parser_meta.get('fallback_used', False):
            self.parsing_failures += 1

        # Enforce bounds
        contribution = max(parameters.MIN_CONTRIBUTION, min(contribution, self.get_stage1_contribution_cap()))

        self.contribution = contribution
        self.contribution_reasoning = llm_reasoning
        self.contribution_facts_used = facts_used
        self.contribution_deepseek_think = deepseek_think
        self.contribution_parser_meta = parser_meta

        self.log_debug(self.round_number, "stage_1_contribution", prompt, response)


    def assign_punishment(self, group_state):
        """
        Decide on assigning punishments or rewards via the LLM.
        """
        prompt = construct_punishment_prompt(self, group_state)
        temperature, top_p = self._persona_sampling_profile("punishment")
        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt,
            response_format={"type": "json_object"},
            max_tokens=2250,
            temperature=temperature,
            top_p=top_p
        )

        punishment_allocations, reward_allocations, reasoning, deanonymized, justifications, facts_used, deepseek_think, parser_meta = parse_punishment_response(response, group_state, self)
        if parser_meta.get('fallback_used', False):
            repair_response = self.api_client.send_request(
                model_name=self.api_client.deployment_name,
                prompt=_schema_repair_prompt(prompt, "Punishment and Reward Choice"),
                response_format={"type": "json_object"},
                max_tokens=2250,
                temperature=temperature,
                top_p=top_p
            )
            retry_vals = parse_punishment_response(repair_response, group_state, self)
            retry_pun, retry_rew, retry_reason, retry_deanon, retry_just, retry_facts, retry_deepseek, retry_meta = retry_vals
            if not retry_meta.get('fallback_used', False):
                response = repair_response
                punishment_allocations, reward_allocations = retry_pun, retry_rew
                reasoning, deanonymized = retry_reason, retry_deanon
                justifications, facts_used, deepseek_think = retry_just, retry_facts, retry_deepseek
                parser_meta = retry_meta

        if parser_meta.get('fallback_used', False):
            self.parsing_failures += 1
        
        self.log_debug(self.round_number, "stage_2_punishment", prompt, response)

        self.punishment_reasoning = reasoning
        self.deanonymized_punishment_reasoning = deanonymized
        self.punishment_justifications = justifications
        self.punishment_facts_used = facts_used
        self.punishment_deepseek_think = deepseek_think
        self.punishment_parser_meta = parser_meta
        self.assigned_punishments = punishment_allocations
        self.assigned_rewards = reward_allocations
        return punishment_allocations, reward_allocations


    def _persona_sampling_profile(self, stage_name):
        """Return sampling settings tuned to the current persona."""
        if self.llm_persona == "RANDOM":
            if stage_name == "punishment":
                return 1.0, 1.0
            return 0.95, 1.0

        if self.llm_persona == "GREEDY":
            return 0.15, 0.8

        return 0.5, 0.95



    def update_payoff(self, amount, is_subsidy=False):
        """
        Update the agent's cumulative and round payoffs.
        Args:
            amount (float): Payoff to add.
            is_subsidy (bool): Whether this payoff is from the subsidy pool.
        """
        if is_subsidy:
            self.last_subsidy += amount
            
        self.round_payoff += amount
        self.cumulative_payoff += amount



    def update_history(self, round_data):
        """
        Record the actions and outcomes of the round.
        Only the most recent round is kept (T-1) since agents now rely on
        their belief_state scratchpad for long-term memory.
        """
        self.history = [round_data]

    def update_beliefs(self, round_feedback, anonymous_data):
        """
        Belief Tracking: ask the LLM to reflect on the round that just
        ended and produce an updated structured belief state.

        This replaces the old sliding-window episodic memory with a
        compact, semantically rich working-memory scratchpad.
        """
        if not getattr(parameters, 'BELIEF_TRACKING_ENABLED', True):
            return

        from core.scenario_config import get_scenario_config
        sc = get_scenario_config(parameters.SCENARIO)

        # Format the anonymous peer data for the reflection prompt
        peer_lines = []
        if anonymous_data:
            for entry in anonymous_data:
                peer_id = entry.get('actual_agent_id', '?')
                peer_lines.append(
                    f"Agent {peer_id}: inst={entry.get('institution_choice', '?')}, "
                    f"contrib={entry.get('contribution', 0)}, "
                    f"group={entry.get('agent_group', 'unknown')}, "
                    f"S1_payoff={entry.get('stage1_payoff', 0):.2f}, "
                    f"S2_payoff={entry.get('stage2_payoff', 0):.2f}"
                )
        peer_block = "\n".join(peer_lines) if peer_lines else "No peer data."

        # Format the agent's own round summary
        own_summary = (
            f"Institution: {round_feedback.get('institution_choice', '?')}, "
            f"Contribution: {round_feedback.get('contribution', 0)}, "
            f"Stage 1 Payoff: {round_feedback.get('stage1_payoff', 0):.2f}, "
            f"Stage 2 Payoff: {round_feedback.get('stage2_payoff', 0):.2f}, "
            f"Total Round Payoff: {round_feedback.get('payoff', 0):.2f}, "
            f"Cumulative Payoff: {round_feedback.get('cumulative_payoff', 0):.2f}, "
            f"Reputation: {round_feedback.get('reputation', 5.0):.1f}"
        )

        import json as _json
        current_belief_str = _json.dumps(self.belief_state, indent=2)

        prompt = f"""You are Agent {self.agent_id} in a {sc['game_name']}. Round {round_feedback.get('round_number', '?')} just ended.

Your task is to UPDATE your internal belief state based on what happened this round.

**Your current belief state (from before this round):**
{current_belief_str}

**Your own results this round:**
{own_summary}

**Peer actions this round:**
{peer_block}

**Instructions:**
- Update your trust assessment for each peer agent based on their behavior.
- Update your institutional strategy based on whether your current approach is working.
- Record any notable observations or patterns you are noticing.
- Be concise: each field should be 1-2 sentences max.

Respond ONLY with a valid JSON object in this exact format:
{{
  "trust_levels": {{
    "<agent_id>": "<1-2 word trust assessment, e.g. 'cooperative', 'free-rider', 'inconsistent', 'trustworthy'>" 
  }},
  "institutional_strategy": "<1-2 sentences on your plan for the next round>",
  "observations": "<1-2 sentences on patterns or trends you notice>"
}}"""

        try:
            response = self.api_client.send_request(
                model_name=self.api_client.deployment_name,
                prompt=prompt,
                max_tokens=768,
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            parsed = robust_json_loads(response)
            if parsed and isinstance(parsed, dict):
                # Validate required keys exist, otherwise keep old state
                if 'trust_levels' in parsed and 'institutional_strategy' in parsed:
                    self.belief_state = {
                        'trust_levels': parsed.get('trust_levels', self.belief_state.get('trust_levels', {})),
                        'institutional_strategy': str(parsed.get('institutional_strategy', self.belief_state.get('institutional_strategy', ''))),
                        'observations': str(parsed.get('observations', self.belief_state.get('observations', '')))
                    }
            self.log_debug(round_feedback.get('round_number', 0), "belief_update", prompt, response)
            round_number = round_feedback.get('round_number', '?')
            strategy = self.belief_state.get('institutional_strategy', '')
            trust_levels = self.belief_state.get('trust_levels', {}) or {}
            trust_preview = ", ".join(sorted(map(str, trust_levels.keys()))[:3])
            trust_suffix = f"; peers={trust_preview}" if trust_preview else ""
            logger.debug(
                f"[Belief Update] Agent {self.agent_id} round {round_number}: "
                f"belief state updated (strategy={strategy!r}{trust_suffix})"
            )
        except Exception as e:
            # On failure, keep the previous belief state unchanged
            logger.warning(f"[Belief Update] Agent {self.agent_id} belief update failed: {e}")

    def reset_for_new_round(self):
        """
        Reset variables that are specific to a round.
        """
        # Move current round data to anonymous data history before resetting
        if hasattr(self, 'current_round_anonymous_data') and self.current_round_anonymous_data is not None:
            round_data = {
                'round_number': self.round_number,
                'anonymous_data': self.current_round_anonymous_data
            }
            self.anonymous_data_history.append(round_data)
            # Ensure the history does not exceed DISPLAY_PAST_ACTIONS
            if len(self.anonymous_data_history) > parameters.DISPLAY_PAST_ACTIONS:
                self.anonymous_data_history.pop(0)
            self.current_round_anonymous_data = None

        # Phase 4 Curiosity: snapshot choices BEFORE they are reset
        if self.institution_choice:
            self.history_institutions.append(self.institution_choice)
        self.history_contributions.append(self.contribution)
        if len(self.history_contributions) > 10:
            self.history_contributions.pop(0)

        # Reset other attributes
        self.contribution = 0
        self.received_punishments = 0
        self.received_rewards = 0
        self.assigned_punishments = {}
        self.assigned_rewards = {}
        self.current_group = None
        self.round_payoff = 0  # Reset current round's payoff

        # Reset reasoning attributes
        self.institution_reasoning = ''
        self.contribution_reasoning = ''
        self.punishment_reasoning = ''
        self.deanonymized_punishment_reasoning = ''
        self.institution_deepseek_think = ''
        self.contribution_deepseek_think = ''
        self.punishment_deepseek_think = ''
        self.anonymized_id_mapping = {}
        self.last_subsidy = 0 # Reset for new round
        self.climate_damage_taken_round = 0.0
        self.ldf_contribution_round = 0.0
        self.ldf_payout_round = 0.0
        self.net_climate_transfer_round = 0.0
        self.institution_parser_meta = {}
        self.contribution_parser_meta = {}
        self.punishment_parser_meta = {}

    def receive_punishment(self, amount):
        """
        Record the amount of punishment received.
        Args:
        amount (float): The total punishment effect received.
        """
        self.received_punishments += amount

    def receive_reward(self, amount):
        """
        Record the amount of reward received.
        Args:
        amount (float): The total reward effect received.
        """
        self.received_rewards += amount

    def get_stage1_payoff(self, group_size, total_group_contribution):
        """
        Calculate the payoff from Stage 1.
        Args:
        group_size (int): The number of members in the group.
        total_group_contribution (float): The sum of contributions in the group.
        Returns:
        float: The payoff from Stage 1.
        """
        if group_size > 0:
            earnings_from_public_good = (parameters.PUBLIC_GOOD_MULTIPLIER * total_group_contribution) / group_size
        else:
            earnings_from_public_good = 0
            
        if self._uses_climate_budget():
            # In climate mode, return the net profit/loss
            stage1_payoff = earnings_from_public_good - self.contribution
        else:
            contribution_cap = self.get_stage1_contribution_cap()
            tokens_kept = contribution_cap - self.contribution
            stage1_payoff = tokens_kept + earnings_from_public_good
            
        return stage1_payoff

    def get_stage2_budget(self):
        """
        Get the Stage 2 budget dynamically.
        In climate mode, it is 5% of their wealth (capped at least at ENDOWMENT_STAGE_2).
        In standard mode, it is ENDOWMENT_STAGE_2.
        """
        if self._uses_climate_budget():
            return max(parameters.ENDOWMENT_STAGE_2, int(self.wealth * 0.05))
        return parameters.ENDOWMENT_STAGE_2

    def get_max_punishment_tokens(self):
        """
        Get the maximum punishment tokens assignable to a single target dynamically.
        In climate mode, it is 5% of their wealth (capped at least at MAX_PUNISHMENT_TOKENS).
        In standard mode, it is MAX_PUNISHMENT_TOKENS.
        """
        if self._uses_climate_budget():
            return max(parameters.MAX_PUNISHMENT_TOKENS, int(self.wealth * 0.05))
        return parameters.MAX_PUNISHMENT_TOKENS

    def get_stage2_payoff(self):
        """
        Calculate the net payoff from Stage 2, after considering assigned punishments and rewards.
        Returns:
        float: The payoff from Stage 2.
        """
        # Tokens used for assigning punishments and rewards
        tokens_spent = (
            sum(self.assigned_punishments.values()) * parameters.PUNISHMENT_COST +
            sum(self.assigned_rewards.values()) * parameters.REWARD_COST
        )

        # Effects of punishments and rewards received
        punishment_effect = self.received_punishments  # Already includes the punishment effect
        reward_effect = self.received_rewards  # Already includes the reward effect

        if self._uses_climate_budget():
            # In climate mode, return the net profit/loss (no free endowment, costs deducted from wealth)
            stage2_payoff = -tokens_spent + reward_effect - punishment_effect
        else:
            # Tokens remaining from the initial Stage 2 endowment
            tokens_remaining = parameters.ENDOWMENT_STAGE_2 - tokens_spent
            stage2_payoff = tokens_remaining + reward_effect - punishment_effect
        return stage2_payoff

    def __repr__(self):
        return f"Agent({self.agent_id}, Cumulative Payoff: {self.cumulative_payoff})"

    def log_debug(self, round_num, stage_name, prompt, response):
        """Helper to save LLM interactions for debugging."""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        filename = f"agent_{self.agent_id}_round_{round_num}_{stage_name}.json"
        with open(os.path.join(log_dir, filename), 'w') as f:
            json.dump({'prompt': prompt, 'response': response}, f, indent=2)
