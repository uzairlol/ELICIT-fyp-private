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

Audits are batched: one LLM call per evaluating agent scores all peers at once.
"""

import logging
import parameters
from scenario_config import get_scenario_config
from utils import robust_json_loads

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
        behavioural consistency this round in a single batched LLM call.

        Returns:
            dict: {other_agent_id (int): {'score': float, 'reasoning': str}}
        """
        targets = [a for a in all_agents if a.agent_id != evaluating_agent.agent_id]
        if not targets:
            return {}

        prompt = self._build_batch_prompt(evaluating_agent, targets, round_number)
        response = self.api_client.send_request(
            model_name=self.api_client.deployment_name,
            prompt=prompt,
            max_tokens=768,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        scores = self._parse_batch_response(response, targets)

        if self._batch_needs_repair(scores, targets):
            repair_prompt = (
                f"{prompt}\n\n"
                "IMPORTANT RETRY (ToM Batch Audit): Your previous response did not include "
                "all required peer scores. Return ONLY one valid JSON object with a "
                '"scores" object containing every listed Agent label exactly once.'
            )
            repair_response = self.api_client.send_request(
                model_name=self.api_client.deployment_name,
                prompt=repair_prompt,
                max_tokens=768,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            repaired = self._parse_batch_response(repair_response, targets)
            if not self._batch_needs_repair(repaired, targets):
                response = repair_response
                scores = repaired

        if not hasattr(evaluating_agent, 'tom_audit_log'):
            evaluating_agent.tom_audit_log = []

        for target in targets:
            entry = scores.get(target.agent_id, {'score': 5.0, 'reasoning': 'Missing score.'})
            score = entry['score']
            reasoning = entry['reasoning']
            logger.debug(
                f"[ToM Audit] Agent {evaluating_agent.agent_id} scored Agent {target.agent_id}: "
                f"{score:.1f}/10 - \"{reasoning[:150]}...\""
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

    def _build_batch_prompt(self, evaluator, targets, round_number):
        sc = get_scenario_config(parameters.SCENARIO)
        currency_name = sc['currency_name']
        targets = sorted(targets, key=lambda agent: agent.agent_id)

        peer_lines = []
        score_template_lines = []
        for target in targets:
            label = f"Agent {target.agent_id}"
            stated_intent = self._truncate(target.contribution_reasoning)
            contribution = target.contribution
            endowment = (
                target.get_stage1_contribution_cap()
                if hasattr(target, 'get_stage1_contribution_cap')
                else parameters.ENDOWMENT_STAGE_1
            )
            peer_lines.append(
                f"- {label}: stated intent = \"{stated_intent}\"; "
                f"contributed {contribution} / {endowment} {currency_name}"
            )
            score_template_lines.append(
                f'    "{label}": {{"trust_score": <integer 1-10>, "reasoning": "<one sentence>"}}'
            )

        peers_block = "\n".join(peer_lines)
        scores_template = ",\n".join(score_template_lines)

        return f"""You are Agent {evaluator.agent_id} in a public goods experiment (Round {round_number}).

You are auditing the behavioral consistency of every other participant this round.

For each peer below, compare their stated intent before contributing with what they actually contributed:
- 10 = action perfectly matched stated intent (highly consistent)
- 1 = action was inconsistent with their words (highly varied)
- 5 = neutral / insufficient data

In each reasoning string, do NOT use specific Agent IDs or numbers. Refer to the individual as "the target" or "this participant".

**Peers to audit:**
{peers_block}

Respond ONLY with valid JSON in this exact format:
{{
  "scores": {{
{scores_template}
  }}
}}

You MUST include every listed Agent label exactly once under "scores".
"""

    def _parse_batch_response(self, response, targets):
        """Parse batched trust scores. Missing peers default to neutral 5.0."""
        results = {}
        try:
            data = robust_json_loads(response)
            scores_block = data.get('scores', {})
            if not isinstance(scores_block, dict):
                if 'trust_score' in data and len(targets) == 1:
                    scores_block = {f"Agent {targets[0].agent_id}": data}
                else:
                    raise ValueError('Expected a JSON object with a "scores" mapping.')

            for target in targets:
                label = f"Agent {target.agent_id}"
                entry = (
                    scores_block.get(label)
                    or scores_block.get(str(target.agent_id))
                    or scores_block.get(target.agent_id)
                )
                score, reasoning = self._parse_score_entry(entry)
                results[target.agent_id] = {'score': score, 'reasoning': reasoning}

        except Exception as e:
            logger.warning(f"[ToM Batch] Parse failed for evaluator batch: {e}")
            for target in targets:
                results[target.agent_id] = {
                    'score': 5.0,
                    'reasoning': f'Batch parsing failed — defaulting to neutral score. {e}',
                }

        return results

    @staticmethod
    def _parse_score_entry(entry):
        if isinstance(entry, dict):
            score = float(entry.get('trust_score', 5))
            reasoning = str(entry.get('reasoning', '') or '')
        elif entry is not None:
            score = float(entry)
            reasoning = ''
        else:
            return 5.0, 'Missing from batch response — defaulting to neutral score.'

        score = max(1.0, min(10.0, score))

        deepseek_thought = ''
        if isinstance(entry, dict):
            deepseek_thought = entry.get('deepseek_thought', '') or ''
        if deepseek_thought:
            reasoning = f"<think>\n{deepseek_thought}\n</think>\n{reasoning}"

        return score, reasoning

    @staticmethod
    def _batch_needs_repair(scores, targets):
        if len(scores) != len(targets):
            return True
        for target in targets:
            entry = scores.get(target.agent_id)
            if not entry:
                return True
            reasoning = str(entry.get('reasoning', '') or '')
            if reasoning.startswith('Missing from batch response') or reasoning.startswith('Batch parsing failed'):
                return True
        return False
