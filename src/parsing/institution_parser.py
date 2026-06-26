import re
from parsing.response_parsing_utils import _unwrap_response_data, _make_parser_meta


def _extract_institution_choice(data):
    institution_choice = str(data.get('institution_choice', '')).upper()
    if not institution_choice:
        institution_choice = str(data.get('institution', '')).upper()
    if institution_choice and institution_choice not in ('SI', 'SFI'):
        embedded = re.findall(r'\bSFI\b|\bSI\b', institution_choice)
        if embedded:
            institution_choice = embedded[-1]
    if not institution_choice:
        decision = str(data.get('decision', '')).upper()
        if 'SFI' in decision:
            institution_choice = 'SFI'
        elif 'SI' in decision:
            institution_choice = 'SI'
    return institution_choice if institution_choice in ('SI', 'SFI') else ''


def parse_institution_choice_response(response, agent_id):
    """
    Parse the LLM's response to extract the institution choice and reasoning.
    Returns: (choice, reasoning, facts_used, deepseek_think, parser_meta)
    """
    expected_keys = ['institution_choice', 'institution', 'decision', 'reasoning', 'facts_used', 'deepseek_think', 'deepseek_thought']
    try:
        data = _unwrap_response_data(response)
        institution_choice = _extract_institution_choice(data)
        reasoning = data.get('reasoning', '')
        deepseek_thought = data.get('deepseek_thought', '')
        if deepseek_thought:
            reasoning = f"<think>\n{deepseek_thought}\n</think>\n" + reasoning
        deepseek_think = data.get('deepseek_think', '')
        facts_used = data.get('facts_used', [])
        if institution_choice in ('SI', 'SFI'):
            return institution_choice, reasoning, facts_used, deepseek_think, _make_parser_meta(data, expected_keys)
        return '', reasoning, facts_used, deepseek_think, _make_parser_meta(data, expected_keys, True, 'Missing/invalid institution choice')
    except Exception as e:
        return '', '', [], '', _make_parser_meta({}, expected_keys, True, f'Institution parse exception: {e}')
