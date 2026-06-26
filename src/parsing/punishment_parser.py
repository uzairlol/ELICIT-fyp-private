import logging
import re
from core import parameters
from parsing.response_parsing_utils import _unwrap_response_data, _apply_stage2_allocations, _make_parser_meta, deanonymize_reasoning

logger = logging.getLogger(__name__)


def parse_punishment_response(response, group_state, agent):
    """
    Parse the LLM's response to extract punishment and reward allocations and reasoning.
    Returns: (punishment_allocations, reward_allocations, reasoning, deanonymized_reasoning, justifications, facts_used, deepseek_think, parser_meta)
    """
    expected_keys = ['punishments', 'rewards', 'reasoning', 'facts_used', 'justifications', 'deepseek_think', 'deepseek_thought']
    try:
        data = _unwrap_response_data(response)

        punishments = data.get('punishments', {}) or {}
        rewards = data.get('rewards', {}) or {}
        reasoning = data.get('reasoning', '')
        deepseek_thought = data.get('deepseek_thought', '')
        if deepseek_thought:
            reasoning = f"<think>\n{deepseek_thought}\n</think>\n" + reasoning
        deepseek_think = data.get('deepseek_think', '')
        justifications = data.get('justifications', {}) or {}
        facts_used = data.get('facts_used', []) or []

        def _expected_target_labels(group_state, agent):
            members = list((group_state or {}).get('members', []) or [])
            others = sorted([member for member in members if getattr(member, 'agent_id', None) != agent.agent_id], key=lambda member: member.agent_id)
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

        expected_labels = _expected_target_labels(group_state, agent)
        repaired_missing_targets = False
        for label in expected_labels:
            if label not in punishments:
                punishments[label] = 0
                repaired_missing_targets = True

        # Deanonymize the reasoning for logging purposes
        deanonymized_reasoning = deanonymize_reasoning(reasoning, agent.anonymized_id_mapping)

        budget = agent.get_stage2_budget() if hasattr(agent, 'get_stage2_budget') else parameters.ENDOWMENT_STAGE_2
        max_per_target = agent.get_max_punishment_tokens() if hasattr(agent, 'get_max_punishment_tokens') else int(getattr(parameters, 'MAX_PUNISHMENT_TOKENS', parameters.ENDOWMENT_STAGE_2))

        punishment_allocations, reward_allocations = _apply_stage2_allocations(
            punishments, rewards, agent, agent.anonymized_id_mapping, group_state, budget, max_per_target
        )

        fallback_used = not bool(punishments)
        fallback_reason = 'Missing punishments object' if fallback_used else ''
        parser_meta = _make_parser_meta(data, expected_keys, fallback_used, fallback_reason)
        parser_meta['repaired_missing_targets'] = repaired_missing_targets
        parser_meta['expected_target_labels'] = expected_labels
        parser_meta['parsed_punishment_labels'] = list(punishments.keys())
        parser_meta['parsed_reward_labels'] = list(rewards.keys())
        return punishment_allocations, reward_allocations, reasoning, deanonymized_reasoning, justifications, facts_used, deepseek_think, parser_meta

    except Exception as e:
        logger.warning(f"Error parsing punishment response: {e}")
        return {}, {}, 'Parsing failed', 'Parsing failed', {}, [], '', _make_parser_meta({}, expected_keys, True, f'Punishment parse exception: {e}')
