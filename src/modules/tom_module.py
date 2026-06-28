# tom_module.py

"""
Theory of Mind (ToM) Module for ELICIT (Emergent LLM Institutions for Climate and International Treaties).

After each round, each agent silently "audits" every other agent for behavioural
consistency (hypocrisy detection):
  - Stated Intent  : the reasoning the agent gave *before* contributing
  - Objective Action: the actual contribution they made

The LLM scores each agent pair on a trustworthiness scale of 1-10.
These scores are stored on each Agent as `tom_scores` and averaged into a
`reputation` value, which feeds back into the RL observation vector and
future LLM prompts, closing the loop between social judgement and behaviour.

Audits are pairwise: one LLM call per evaluator-target pair.
"""

import logging
from core import parameters
from core.scenario_config import get_scenario_config
from core.utils import robust_json_loads

logger = logging.getLogger(__name__)


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
        behavioural consistency this round via one LLM call per peer.

        Returns:
            dict: {other_agent_id (int): {'score': float, 'reasoning': str}}
        """
        scores = {}
        if not hasattr(evaluating_agent, 'tom_audit_log'):
            evaluating_agent.tom_audit_log = []

        for target in all_agents:
            if target.agent_id == evaluating_agent.agent_id:
                continue

            score, reasoning = self._score_agent(
                evaluator=evaluating_agent,
                target=target,
                round_number=round_number,
            )
            if score is None:
                continue

            scores[target.agent_id] = {'score': score, 'reasoning': reasoning}

            if getattr(parameters, 'TOM_VERBOSE', False):
                logger.info(
                    f"[ToM] Agent {evaluating_agent.agent_id} scored Agent {target.agent_id}: "
                    f"{score:.1f}/10"
                )
            else:
                logger.debug(
                    f"[ToM Audit] Agent {evaluating_agent.agent_id} scored Agent {target.agent_id}: "
                    f"{score:.1f}/10"
                )

            evaluating_agent.tom_audit_log.append({
                'round': round_number,
                'target_agent': target.agent_id,
                'trust_score': score,
                'reasoning': reasoning,
            })

        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text, limit=180):
        text = (text or "No reasoning provided.").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _score_agent(self, evaluator, target, round_number):
        """Score one peer. Returns (score, reasoning) or (None, '') on failure."""
        prompt = self._build_pair_prompt(evaluator, target, round_number)
        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt,
            max_tokens=128,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        score, parse_error = self._parse_score_response(response)
        if parse_error:
            repair_prompt = (
                f"{prompt}\n\n"
                "IMPORTANT RETRY (ToM Audit): Return ONLY one valid JSON object with "
                '"trust_score" as an integer from 1 to 10. No extra text.'
            )
            repair_response = self.api_client.send_request(
                model_name=self.api_client.deployment_name,
                prompt=repair_prompt,
                max_tokens=128,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            score, parse_error = self._parse_score_response(repair_response)
            if not parse_error:
                response = repair_response

        if parse_error:
            logger.warning(
                f"[ToM] Agent {evaluator.agent_id} failed to score Agent {target.agent_id}: "
                f"{parse_error}"
            )
            return None, ''

        return score, ''

    def _build_pair_prompt(self, evaluator, target, round_number):
        sc = get_scenario_config(parameters.SCENARIO)
        currency_name = sc['currency_name']
        stated_intent = self._truncate(target.contribution_reasoning)
        contribution = target.contribution
        endowment = (
            target.get_stage1_contribution_cap()
            if hasattr(target, 'get_stage1_contribution_cap')
            else parameters.ENDOWMENT_STAGE_1
        )

        return f"""You are Agent {evaluator.agent_id} in a public goods experiment (Round {round_number}).

Task: Score the behavioral consistency of Agent {target.agent_id} this round.

**Stated intent before contributing:**
"{stated_intent}"

**Actual contribution:**
{contribution} / {endowment} {currency_name}

**Scoring scale:**
- 10 = action perfectly matched stated intent
- 5 = neutral / insufficient data
- 1 = action inconsistent with stated intent

**Required JSON shape:**
{{
  "trust_score": 5
}}

**FINAL OUTPUT RULES:**
- trust_score MUST be an integer from 1 to 10.
- Return exactly ONE JSON object. No other keys. No text outside the JSON."""

    def _parse_score_response(self, response):
        """Parse a single trust score. Returns (score, error_message)."""
        try:
            data = robust_json_loads(response)
            if 'trust_score' not in data:
                raise ValueError('Missing trust_score')
            score = float(data['trust_score'])
            score = max(1.0, min(10.0, score))
            return score, ''
        except Exception as e:
            return None, str(e)
