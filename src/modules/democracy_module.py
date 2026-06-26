# democracy_module.py

"""
Democracy Module for ELICIT (Emergent LLM Institutions for Climate and International Treaties).

Every DEMOCRACY_INTERVAL rounds a "constitutional moment" is triggered:

  Phase A — Proposals:
    Each SI-member agent proposes ONE specific rule change from a bounded
    whitelist (to prevent runaway drift).  Example outputs:
      { "rule": "PUNISHMENT_EFFECT", "new_value": 4, "reason": "..." }

  Phase B — Voting:
    Each agent is shown all collected proposals and casts a single vote
    (by proposal index).  The majority-winning proposal is applied to the
    live `parameters` object immediately.

The winning proposal and vote tallies are returned so `environment.py` can
log them in the round data / results JSON.
"""

import logging
from core import parameters
import random
import json
import re
from core.utils import robust_json_loads

logger = logging.getLogger(__name__)

# Only these parameters may be changed by democratic vote.
# This prevents agents from corrupting SEED, NUM_AGENTS, NUM_ROUNDS, etc.
GENERAL_DEMOCRACY_PARAMS = {
    'PUNISHMENT_EFFECT',
    'REWARD_EFFECT',
    'ENDOWMENT_STAGE_2',
    'MAX_PUNISHMENT_TOKENS',
    'SUBSIDY_FRACTION',
    'SUBSIDY_TOP_N',
}

LDF_DEMOCRACY_PARAMS = {
    'LDF_PAYOUT_DAMAGE_WEIGHT',
    'LDF_MAX_COVERAGE',
    'LDF_EQUITY_WEIGHT',
}

def get_allowed_democracy_params():
    allowed = set(GENERAL_DEMOCRACY_PARAMS)
    if getattr(parameters, 'LDF_ENABLED', False) or getattr(parameters, 'SCENARIO', '').lower() == 'ldf':
        allowed.update(LDF_DEMOCRACY_PARAMS)
    return allowed


class DemocracyModule:
    """
    Runs a proposal-then-vote constitutional session among SI agents.
    """

    def __init__(self, api_client):
        self.api_client = api_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_constitutional_session(self, agents, round_number, oracle=None):
        """
        Full proposal + vote cycle.

        Args:
            agents: list of Agent instances (all agents, SI + SFI)
            round_number (int): The round that just ended.
            oracle (Oracle, optional): Welfare predictor for annotating proposals.

        Returns:
            dict with keys:
              'proposals'      : list of proposal dicts
              'votes'          : dict {agent_id: proposal_index}
              'winning_proposal': the winning proposal dict (or None)
              'applied'        : bool — whether the change was applied
        """
        logger.info(f"--- Constitutional Moment (Round {round_number}) ---")

        # Phase A: Collect proposals from every agent
        proposals = self._collect_proposals(agents, round_number)

        if not proposals:
            logger.info("No valid proposals received. Skipping vote.")
            return {
                'proposals': [],
                'votes': {},
                'winning_proposal': None,
                'applied': False
            }

        # Phase B: Vote (pass oracle annotations so agents see welfare predictions)
        oracle_annotations = oracle.annotate_proposals(proposals) if oracle else None
        votes = self._collect_votes(agents, proposals, round_number, oracle_annotations=oracle_annotations)

        # Tally
        winning_proposal, tally = self._tally_votes(proposals, votes)

        logger.info(f"Vote tally: {tally}")

        # Apply winning rule
        applied = False
        if winning_proposal:
            applied = self._apply_rule(winning_proposal)
            if applied:
                logger.info(
                    f"Rule applied: {winning_proposal['rule']} → {winning_proposal['new_value']}"
                )

        logger.info(f"--- End of Constitutional Session ---")

        return {
            'proposals': proposals,
            'votes': votes,
            'tally': tally,
            'winning_proposal': winning_proposal,
            'applied': applied
        }

    # ------------------------------------------------------------------
    # Phase A: Proposals
    # ------------------------------------------------------------------

    def _build_tuneable_str(self):
        """Show whitelisted numeric parameters with descriptions, randomized to prevent position bias."""
        import random
        
        param_descriptions = {
            'PUNISHMENT_EFFECT': 'Multiplier applied to target\'s payoff when punished (e.g., 3 means losing 3 tokens per 1 token spent).',
            'REWARD_EFFECT': 'Multiplier applied to target\'s payoff when rewarded.',
            'ENDOWMENT_STAGE_2': 'Number of tokens agents are given specifically for punishing/rewarding others.',
            'MAX_PUNISHMENT_TOKENS': 'Maximum number of punishment tokens one agent can assign to a single target.',
            'SUBSIDY_FRACTION': 'Fraction of SI punishment costs pooled and redistributed to top SI contributors as a subsidy.',
            'SUBSIDY_TOP_N': 'Number of top contributors in the SI who share the subsidy pool.',
            'LDF_PAYOUT_DAMAGE_WEIGHT': 'How strongly direct damage affects LDF payouts after shocks.',
            'LDF_MAX_COVERAGE': 'Maximum fraction of an agent\'s climate damage covered by LDF payouts.',
            'LDF_EQUITY_WEIGHT': 'How strongly the LDF prioritizes poorer developing nations. 0.0 ignores wealth, higher values heavily skew payouts to the poorest.'
        }
        
        lines = []
        # Randomize order to prevent LLM position bias (anchoring on the first item)
        param_names = list(get_allowed_democracy_params())
        random.shuffle(param_names)
        
        for name in param_names:
            val = getattr(parameters, name, None)
            desc = param_descriptions.get(name, "")
            if val is not None:
                lines.append(f"  - {name}: current value = {val} | Effect: {desc}")
        return "\n".join(lines)

    def _collect_proposals(self, agents, round_number):
        """Ask each agent to propose one rule change."""
        proposals = []
        seen_proposals = set() # (rule, new_value) pairs to filter exact duplicates

        tuneable_str = self._build_tuneable_str()

        # Shuffle agents to prevent priority bias
        agents_copy = list(agents)
        random.shuffle(agents_copy)

        for agent in agents_copy:
            # Phase 4 Curiosity: suggest a parameter they haven't explored yet
            curiosity_hint = ""
            if parameters.CURIOSITY_ENABLED and parameters.CURIOSITY_BONUS_PROMPT:
                all_params = list(get_allowed_democracy_params())
                unexplored = [p for p in all_params if p not in agent.explored_params]
                if unexplored:
                    target = random.choice(unexplored)
                    curiosity_hint = f"\n**Note:** You haven't proposed a change to `{target}` in your history. Rule changes can provide data on different governance outcomes."

            proposal = self._get_proposal(agent, round_number, tuneable_str, curiosity_hint=curiosity_hint)
            
            # Phase 4: Dynamic Validation
            validated = self._validate_proposal(proposal)
            if validated:
                rule = validated['rule']
                new_val = validated['new_value']
                validated['proposer'] = agent.agent_id

                # Deduplicate: only skip if it's the EXACT same rule and value
                # This allows different agents to suggest different values for the same rule.
                proposal_key = (rule, new_val)
                if proposal_key not in seen_proposals:
                    proposals.append(validated)
                    seen_proposals.add(proposal_key)
                    agent.explored_params.add(rule) # Mark as explored
                    logger.info(
                        f"Agent {agent.agent_id} proposes: "
                        f"{rule} → {new_val}  (reason: {validated.get('reason','')[:60]})"
                    )
        return proposals

    def _validate_proposal(self, proposal):
        if not proposal or not isinstance(proposal, dict):
            return None
            
        rule = proposal.get('rule')
        new_val = proposal.get('new_value')
        
        # Only allow whitelisted parameters
        if rule not in get_allowed_democracy_params():
            return None
        
        if not hasattr(parameters, rule):
            return None
        
        current = getattr(parameters, rule)
        if not isinstance(current, (int, float)):
            return None
            
        # Ensure correct type (safely convert string representations of floats to int if needed)
        try:
            new_val = type(current)(float(new_val))
        except (ValueError, TypeError):
            return None
            
        # Safety clamp: 10x range
        if current != 0:
            lower = current * 0.1
            upper = current * 10
            new_val = max(lower, min(new_val, upper))
        else:
            new_val = max(0.0, min(new_val, 10.0))
            
        proposal['new_value'] = new_val
        return proposal

    def _get_proposal(self, agent, round_number, tuneable_str, curiosity_hint=""):
        """LLM call: ask one agent to propose a rule change."""
        prompt = f"""You are Agent {agent.agent_id} in a public goods game after Round {round_number}.

The community is holding a constitutional vote. You may propose ONE specific rule change to improve collective welfare.

**Current Parameters of the Simulation:**
{tuneable_str}
{curiosity_hint}

**Current cumulative payoff:** {agent.cumulative_payoff:.0f} tokens

**Your task:**
Propose a single change to one of the numeric parameters listed above. You are not restricted to a subset — chose what you believe will most improve trust and sustainability based on what you have observed in past rounds.

Respond ONLY with valid JSON in this exact format:
{{
  "rule": "<exact parameter name>",
  "new_value": <new numeric value>,
  "reason": "<one sentence justification>"
}}"""

        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt,
            max_tokens=768,
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        return self._parse_json_response(response)

    # ------------------------------------------------------------------
    # Phase B: Voting
    # ------------------------------------------------------------------

    def _collect_votes(self, agents, proposals, round_number, oracle_annotations=None):
        """Ask each agent to vote on the collected proposals."""
        votes = {}

        # Base proposal listing
        proposal_lines = []
        for i, p in enumerate(proposals):
            line = (
                f"  [{i}] {p['rule']} → {p['new_value']} "
                f"(proposed by Agent {p.get('proposer','?')}): {p.get('reason','')}"
            )
            # Append oracle annotation if available
            if oracle_annotations and i < len(oracle_annotations):
                line += f"\n      📊 Oracle: {oracle_annotations[i]}"
            proposal_lines.append(line)

        proposals_str = "\n".join(proposal_lines)

        for agent in agents:
            vote = self._get_vote(agent, proposals_str, len(proposals), round_number)
            if vote is not None:
                votes[agent.agent_id] = vote
                logger.info(f"Agent {agent.agent_id} votes for proposal [{vote}]")

        return votes

    def _get_vote(self, agent, proposals_str, num_proposals, round_number):
        """LLM call: ask one agent to cast a vote."""
        prompt = f"""You are Agent {agent.agent_id} in a public goods game after Round {round_number}.

The community has collected the following rule-change proposals:

{proposals_str}

Vote for the proposal index (0–{num_proposals - 1}) you believe will best improve group welfare.

Respond ONLY with valid JSON:
{{
  "vote": <integer index of your chosen proposal>,
  "reason": "<one sentence>"
}}"""

        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt,
            max_tokens=768,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        parsed = self._parse_json_response(response)
        if parsed:
            try:
                idx = int(parsed.get('vote', -1))
                if 0 <= idx < num_proposals:
                    return idx
            except (ValueError, TypeError):
                pass
        return None

    # ------------------------------------------------------------------
    # Tallying and applying
    # ------------------------------------------------------------------

    def _tally_votes(self, proposals, votes):
        """Return (winning_proposal, tally_dict)."""
        tally = {i: 0 for i in range(len(proposals))}
        for vote_idx in votes.values():
            if vote_idx in tally:
                tally[vote_idx] += 1

        if not tally:
            return None, tally

        max_votes = max(tally.values())
        if max_votes == 0:
            return None, tally

        # Break ties randomly instead of depending on dictionary insertion order
        winners = [idx for idx, votes in tally.items() if votes == max_votes]
        winning_idx = random.choice(winners)

        return proposals[winning_idx], tally

    def _apply_rule(self, proposal):
        """
        Apply the winning proposal to the live parameters module.
        Returns True if successfully applied.
        """
        rule = proposal.get('rule')
        new_value = proposal.get('new_value')
        if rule and hasattr(parameters, rule):
            setattr(parameters, rule, new_value)
            return True
        return False

    # ------------------------------------------------------------------
    # Shared JSON parser
    # ------------------------------------------------------------------

    def _parse_json_response(self, response):
        """Robustly parse a JSON object from an LLM response string."""
        try:
            return robust_json_loads(response)
        except Exception:
            return None
