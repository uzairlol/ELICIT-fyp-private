# tom_module.py

"""
Theory of Mind (ToM) Module for SanctSim.

After each round, each agent silently "audits" every other agent for behavioural
consistency (hypocrisy detection):
  - Stated Intent  : the reasoning the agent gave *before* contributing
  - Objective Action: the actual contribution they made

The LLM scores each agent pair on a trustworthiness scale of 1-10.
These scores are stored on each Agent as `tom_scores` and averaged into a
`reputation` value, which feeds back into the RL observation vector and
future LLM prompts, closing the loop between social judgement and behaviour.
"""

import json
import re
import parameters
from scenario_config import get_scenario_config


class TomModule:
    """
    Performs Theory of Mind audits on behalf of a single evaluating agent.
    """

    def __init__(self, api_client):
        self.api_client = api_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit_round(self, evaluating_agent, all_agents, round_number):
        """
        For the given evaluating_agent, score every *other* agent's
        behavioural consistency this round.

        Returns:
            dict: {other_agent_id (int): trust_score (float 1-10)}
        """
        scores = {}
        for other_agent in all_agents:
            if other_agent.agent_id == evaluating_agent.agent_id:
                continue

            score, reasoning = self._score_agent(
                evaluator=evaluating_agent,
                target=other_agent,
                round_number=round_number
            )
            scores[other_agent.agent_id] = {'score': score, 'reasoning': reasoning}

            # Optionally store the reasoning on the evaluator for logging
            if not hasattr(evaluating_agent, 'tom_audit_log'):
                evaluating_agent.tom_audit_log = []
            evaluating_agent.tom_audit_log.append({
                'round': round_number,
                'target_agent': other_agent.agent_id,
                'trust_score': score,
                'reasoning': reasoning
            })

        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_agent(self, evaluator, target, round_number):
        """
        Ask the LLM to rate the target agent's consistency based on
        the gap between stated intent and actual action.
        """
        stated_intent = target.contribution_reasoning or "No reasoning provided."
        actual_contribution = target.contribution
        endowment = target.get_stage1_contribution_cap() if hasattr(target, 'get_stage1_contribution_cap') else parameters.ENDOWMENT_STAGE_1
        sc = get_scenario_config(parameters.SCENARIO)
        currency_name = sc['currency_name']

        prompt = f"""You are Agent {evaluator.agent_id} in a public goods experiment (Round {round_number}).

You are auditing the behavior of another participant (Agent {target.agent_id}) to assess their behavioral consistency.

**What they said before contributing (Stated Intent):**
"{stated_intent}"

**What they actually contributed:**
{actual_contribution} out of {endowment} {currency_name}.

**Your task:**
Based on alignment between their stated reasoning and their actual action:
- A score of 10 means their action perfectly matched their stated intent (highly consistent).
- A score of 1 means their action completely was inconsistent with their words (highly varied).
- A score of 5 is neutral / insufficient data.

IMPORTANT: In your reasoning string, do NOT use specific Agent IDs or numbers. Refer to the individual simply as "the target" or "this participant".

Respond ONLY with valid JSON in this exact format:
{{
  "trust_score": <integer 1-10>,
  "reasoning": "<one sentence explanation>"
}}"""

        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt,
            max_tokens=768,
            temperature=0.3,   # Low temperature for consistent scoring
            response_format={"type": "json_object"}
        )

        return self._parse_score_response(response)

    def _parse_score_response(self, response):
        """
        Parse the LLM's trust-score JSON response.
        Returns (score: float, reasoning: str).
        """
        try:
            from utils import robust_json_loads
            data = robust_json_loads(response)

            score = float(data.get('trust_score', 5))
            # Clamp to valid range
            score = max(1.0, min(10.0, score))
            reasoning = data.get('reasoning', '')
            
            deepseek_thought = data.get('deepseek_thought', '')
            if deepseek_thought:
                reasoning = f"<think>\n{deepseek_thought}\n</think>\n" + reasoning
                
            return score, reasoning

        except Exception as e:
            return 5.0, f"Parsing failed — defaulting to neutral score. {e}"
