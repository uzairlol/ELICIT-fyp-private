import logging
from core import parameters
from parsing.response_parsing_utils import _unwrap_response_data, _parse_int_safe, _make_parser_meta

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

        if contribution is None:
            raise ValueError('The response did not contain a contribution value.')

        reasoning = data.get('reasoning', '')
        deepseek_thought = data.get('deepseek_thought', '')
        if deepseek_thought:
            reasoning = f"<think>\n{deepseek_thought}\n</think>\n" + reasoning
        deepseek_think = data.get('deepseek_think', '')
        facts_used = data.get('facts_used', [])
        return contribution, reasoning, facts_used, deepseek_think, _make_parser_meta(data, expected_keys)
    except Exception as e:
        logger.warning(f"agent_{agent.agent_id} failed to parse contribution: {e}")
        return None, '', [], '', _make_parser_meta({}, expected_keys, True, f'Contribution parse exception: {e}')
