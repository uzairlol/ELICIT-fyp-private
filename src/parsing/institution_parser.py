import re
import random
from core import parameters
from parsing.response_parsing_utils import _unwrap_response_data, _make_parser_meta


def parse_institution_choice_response(response, agent_id):
    """
    Parse the LLM's response to extract the institution choice and reasoning.
    Returns: (choice, reasoning, facts_used, deepseek_think, parser_meta)
    """
    expected_keys = ['institution_choice', 'institution', 'decision', 'reasoning', 'facts_used', 'deepseek_think', 'deepseek_thought']
    try:
        data = _unwrap_response_data(response)
        institution_choice = str(data.get('institution_choice', '')).upper()
        if not institution_choice:
            institution_choice = str(data.get('institution', '')).upper()
        if institution_choice and institution_choice not in ('SI', 'SFI'):
            # Handle verbose labels like "Group B (Sanctioning Institution - SI)".
            embedded = re.findall(r'\bSFI\b|\bSI\b', institution_choice)
            if embedded:
                institution_choice = embedded[-1]
        if not institution_choice:
            decision = str(data.get('decision', '')).upper()
            if 'SFI' in decision:
                institution_choice = 'SFI'
            elif 'SI' in decision:
                institution_choice = 'SI'
        reasoning = data.get('reasoning', '')
        deepseek_thought = data.get('deepseek_thought', '')
        if deepseek_thought:
            reasoning = f"<think>\n{deepseek_thought}\n</think>\n" + reasoning
        deepseek_think = data.get('deepseek_think', '')
        facts_used = data.get('facts_used', [])
        if institution_choice in ['SI', 'SFI']:
            return institution_choice, reasoning, facts_used, deepseek_think, _make_parser_meta(data, expected_keys)
        else:
            raw_text = str(data.get('__raw_response__') or response or '')
            raw_upper = raw_text.upper()
            if 'SFI' in raw_upper or 'SI' in raw_upper:
                candidates = re.findall(r'\bSFI\b|\bSI\b', raw_upper)
                if candidates:
                    recovered = candidates[-1]
                    return recovered, f"Recovered from free-text output. Raw: {raw_text[:100]}...", [], deepseek_think, _make_parser_meta(data, expected_keys, True, 'Recovered SI/SFI from free text')
            fallback = random.choice(['SI', 'SFI'])
            return fallback, reasoning, facts_used, deepseek_think, _make_parser_meta(data, expected_keys, True, 'Missing/invalid institution choice')
    except Exception as e:
        # Safety fallback: recover explicit SI/SFI from raw text when JSON parsing fails.
        raw_upper = str(response or "").upper()
        if '"INSTITUTION_CHOICE"' in raw_upper:
            if '"SI"' in raw_upper and '"SFI"' not in raw_upper:
                return 'SI', f"Recovered from non-JSON output. Raw: {str(response)[:100]}...", [], '', _make_parser_meta({}, expected_keys, True, 'Recovered SI from non-JSON')
            if '"SFI"' in raw_upper:
                return 'SFI', f"Recovered from non-JSON output. Raw: {str(response)[:100]}...", [], '', _make_parser_meta({}, expected_keys, True, 'Recovered SFI from non-JSON')

        # Loose fallback: if SI/SFI appears in free text, use the last explicit mention.
        candidates = re.findall(r'\bSFI\b|\bSI\b', raw_upper)
        if candidates:
            recovered = candidates[-1]
            if recovered in ('SI', 'SFI'):
                return recovered, f"Recovered from free-text output. Raw: {str(response)[:100]}...", [], '', _make_parser_meta({}, expected_keys, True, 'Recovered SI/SFI from text')

        fallback = random.choice(['SI', 'SFI'])
        return fallback, f"Parsing error (Random Fallback {fallback}). Raw: {str(response)[:100]}...", [], '', _make_parser_meta({}, expected_keys, True, f'Institution parse exception: {e}')
