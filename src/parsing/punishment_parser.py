import logging
import re
from core import parameters
from parsing.response_parsing_utils import (
    _unwrap_response_data,
    _apply_stage2_allocations,
    _make_parser_meta,
    deanonymize_reasoning,
    _normalize_allocation_map,
    _normalize_label_key,
    _parse_allocation_tokens,
    _stage2_total_cost,
)

logger = logging.getLogger(__name__)


def _expected_target_labels(group_state, agent):
    members = list((group_state or {}).get('members', []) or [])
    others = sorted(
        [member for member in members if getattr(member, 'agent_id', None) != agent.agent_id],
        key=lambda member: member.agent_id,
    )
    labels = []
    use_anonymity = bool(getattr(parameters, 'ANONYMITY', False))
    if str(getattr(parameters, 'SCENARIO', '')).lower() == 'climate':
        use_anonymity = False
    if bool(getattr(parameters, 'CLIMATE_SHOCK_ENABLED', False)) or bool(getattr(parameters, 'LDF_ENABLED', False)):
        use_anonymity = False

    for member in others:
        if use_anonymity:
            if hasattr(agent, 'pseudonym_mapping'):
                label_id = agent.pseudonym_mapping.get(member.agent_id, -1)
                if label_id == -1:
                    continue
                labels.append(f'Agent {label_id}')
        else:
            labels.append(f'Agent {member.agent_id}')
    return labels


def _punishment_labels_complete(raw_punishments, expected_labels):
    if not isinstance(raw_punishments, dict):
        return False
    normalized_keys = {_normalize_label_key(key) for key in raw_punishments.keys()}
    return all(label in normalized_keys for label in expected_labels)


def _justifications_complete(raw_justifications, expected_labels):
    if not isinstance(raw_justifications, dict):
        return False
    normalized_keys = {_normalize_label_key(key) for key in raw_justifications.keys()}
    return all(label in normalized_keys for label in expected_labels)


def _reasoning_matches_zero_allocations(reasoning, punishments):
    if not all(_parse_allocation_tokens(v) == 0 for v in punishments.values()):
        return True
    text = str(reasoning or '').lower()
    if not text.strip():
        return False
    explicit_zero = any(
        phrase in text
        for phrase in (
            'all amounts are 0',
            'all punishment amounts are 0',
            'no punishments',
            'punish nobody',
            'punishing nobody',
            'not punishing anyone',
            'zero punishments',
        )
    )
    if explicit_zero:
        return True
    vague_punish_talk = bool(re.search(r'\bpunish(?:ing|ed|ment)?\b', text))
    return not vague_punish_talk


def parse_punishment_response(response, group_state, agent):
    """
    Parse the LLM's response to extract punishment and reward allocations and reasoning.
    Returns: (punishment_allocations, reward_allocations, reasoning, deanonymized_reasoning, justifications, facts_used, deepseek_think, parser_meta)
    """
    expected_keys = ['punishments', 'rewards', 'reasoning', 'facts_used', 'justifications', 'deepseek_think', 'deepseek_thought']
    try:
        data = _unwrap_response_data(response)
        expected_labels = _expected_target_labels(group_state, agent)

        raw_punishments = data.get('punishments')
        raw_justifications = data.get('justifications')
        rewards = data.get('rewards', {}) or {}
        reasoning = data.get('reasoning', '')
        deepseek_thought = data.get('deepseek_thought', '')
        if deepseek_thought:
            reasoning = f"<think>\n{deepseek_thought}\n</think>\n" + reasoning
        deepseek_think = data.get('deepseek_think', '')
        justifications = raw_justifications if isinstance(raw_justifications, dict) else {}
        facts_used = data.get('facts_used', []) or []

        retry_reason = ''
        if not isinstance(raw_punishments, dict):
            retry_reason = 'Missing or invalid punishments object'
        elif not _punishment_labels_complete(raw_punishments, expected_labels):
            retry_reason = 'punishments missing one or more required target labels'
        elif not _justifications_complete(raw_justifications, expected_labels):
            retry_reason = 'justifications missing one or more required target labels'
        elif not str(reasoning or '').strip():
            retry_reason = 'missing reasoning'

        punishments = _normalize_allocation_map(raw_punishments if isinstance(raw_punishments, dict) else {}, expected_labels)
        rewards = _normalize_allocation_map(rewards, [], fill_missing=False) if rewards else {}

        budget = agent.get_stage2_budget() if hasattr(agent, 'get_stage2_budget') else parameters.ENDOWMENT_STAGE_2

        punishment_allocations, reward_allocations = _apply_stage2_allocations(
            punishments, rewards, agent, agent.anonymized_id_mapping, group_state, budget
        )

        total_cost = _stage2_total_cost(punishment_allocations, reward_allocations)
        if not retry_reason and total_cost > budget:
            retry_reason = f'total spend {total_cost} exceeds budget {budget}'

        all_zero_punishments = all(_parse_allocation_tokens(v) == 0 for v in punishments.values())
        semantic_retry = False
        if not retry_reason and all_zero_punishments and not _reasoning_matches_zero_allocations(reasoning, punishments):
            semantic_retry = True

        if not retry_reason:
            for label in expected_labels:
                amount = _parse_allocation_tokens(punishments.get(label, 0))
                if amount > 0 and not str(justifications.get(label, '') or '').strip():
                    retry_reason = f'missing justification for {label}'
                    break

        deanonymized_reasoning = deanonymize_reasoning(reasoning, agent.anonymized_id_mapping)

        parser_meta = _make_parser_meta(data, expected_keys, bool(retry_reason), retry_reason)
        parser_meta['expected_target_labels'] = expected_labels
        parser_meta['parsed_punishment_labels'] = list(punishments.keys())
        parser_meta['parsed_reward_labels'] = list(rewards.keys())
        parser_meta['all_zero_punishments'] = all_zero_punishments
        parser_meta['semantic_retry'] = semantic_retry
        parser_meta['raw_punishment_values'] = {k: _parse_allocation_tokens(v) for k, v in punishments.items()}
        parser_meta['total_spend'] = total_cost
        parser_meta['budget'] = budget

        if retry_reason:
            return {}, {}, reasoning, deanonymized_reasoning, justifications, facts_used, deepseek_think, parser_meta

        return punishment_allocations, reward_allocations, reasoning, deanonymized_reasoning, justifications, facts_used, deepseek_think, parser_meta

    except Exception as e:
        logger.warning(f"Error parsing punishment response: {e}")
        return {}, {}, '', '', {}, [], '', _make_parser_meta({}, expected_keys, True, f'Punishment parse exception: {e}')
