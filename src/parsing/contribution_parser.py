import logging
import re
from core import parameters
from parsing.response_parsing_utils import _unwrap_response_data, _parse_int_safe, _make_parser_meta
from core.utils import uses_climate_budget

logger = logging.getLogger(__name__)


def parse_contribution_response_v2(response, agent):
    """
    Parse the LLM's response to extract BOTH contribution amount and reasoning.
    Returns: (contribution, reasoning, facts_used, deepseek_think, parser_meta)
    """
    expected_keys = ['contribution', 'contribution_to_project', 'tokens_contributed', 'contribution_amount', 'reasoning', 'facts_used', 'deepseek_think', 'deepseek_thought']
    try:
        data = _unwrap_response_data(response)

        contribution = None
        for key in ('contribution', 'contribution_to_project', 'tokens_contributed', 'contribution_amount'):
            if key in data:
                contribution = _parse_int_safe(data.get(key))
                break

        raw_text = str(data.get('__raw_response__') or response or '')
        if contribution is None:
            # try labelled key=value in raw text
            contribution_match = re.search(r'"(?:contribution|contribution_to_project|tokens_contributed|contribution_amount)"\s*[:=]\s*"?(-?\d+)"?', raw_text, flags=re.IGNORECASE)
            if not contribution_match:
                contribution_match = re.search(r'\bcontribut(?:e|ion)\b[^\d-]{0,20}(-?\d+)', raw_text, flags=re.IGNORECASE)
            if not contribution_match:
                numeric_candidates = [int(match) for match in re.findall(r'(?<!\d)(-?\d+)(?!\.\d)', raw_text)]
                plausible_candidates = [value for value in numeric_candidates if parameters.MIN_CONTRIBUTION <= value <= agent.get_stage1_contribution_cap()]
                if len(plausible_candidates) == 1:
                    contribution_match = plausible_candidates[0]
            if not contribution_match:
                raise ValueError("The response did not contain a recoverable contribution value.")
            if isinstance(contribution_match, int):
                contribution = contribution_match
            else:
                contribution = int(contribution_match.group(1))

        contribution = max(parameters.MIN_CONTRIBUTION, min(contribution, agent.get_stage1_contribution_cap()))

        reasoning = data.get('reasoning', '')
        deepseek_thought = data.get('deepseek_thought', '')
        if deepseek_thought:
            reasoning = f"<think>\n{deepseek_thought}\n</think>\n" + reasoning
        deepseek_think = data.get('deepseek_think', '')
        facts_used = data.get('facts_used', [])
        return contribution, reasoning, facts_used, deepseek_think, _make_parser_meta(data, expected_keys)
    except Exception as e:
        raw_text = str(response or "")
        # Safety fallback: recover contribution number from non-JSON text.
        contribution_match = re.search(r'"(?:contribution|contribution_to_project|tokens_contributed|contribution_amount)"\s*[:=]\s*"?(-?\d+)"?', raw_text, flags=re.IGNORECASE)
        if not contribution_match:
            contribution_match = re.search(r'\bcontribut(?:e|ion)\b[^\d-]{0,20}(-?\d+)', raw_text, flags=re.IGNORECASE)
        if not contribution_match:
            numeric_candidates = [int(match) for match in re.findall(r'(?<!\d)(-?\d+)(?!\.\d)', raw_text)]
            plausible_candidates = [value for value in numeric_candidates if parameters.MIN_CONTRIBUTION <= value <= agent.get_stage1_contribution_cap()]
            if len(plausible_candidates) == 1:
                contribution_match = plausible_candidates[0]
        if contribution_match:
            recovered_val = contribution_match if isinstance(contribution_match, int) else int(contribution_match.group(1))
            recovered_val = max(parameters.MIN_CONTRIBUTION, min(recovered_val, agent.get_stage1_contribution_cap()))
            return recovered_val, f"Recovered from non-JSON output. Raw: {raw_text[:100]}...", [], '', _make_parser_meta({}, expected_keys, True, 'Recovered contribution from text')

        if uses_climate_budget():
            fallback_val = max(parameters.MIN_CONTRIBUTION, int(getattr(agent, 'wealth', parameters.ENDOWMENT_STAGE_1)) // 2)
        else:
            fallback_val = parameters.ENDOWMENT_STAGE_1 // 2
        logger.warning(f"agent_{agent.agent_id} failed to parse contribution: {e}. Using fallback {fallback_val}")
        return fallback_val, f"Lazy Fallback due to parsing error. Raw: {response[:100]}...", [], '', _make_parser_meta({}, expected_keys, True, f'Contribution parse exception: {e}')
