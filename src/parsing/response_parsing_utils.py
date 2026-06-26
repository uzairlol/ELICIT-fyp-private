#response_parsing_utils.py

import logging
import re
from core import parameters
from core.utils import robust_json_loads

logger = logging.getLogger(__name__)


def _unwrap_response_data(response):
    """Return a dict-like parsed object with '__raw_response__' preserved.

    Handles dict inputs, JSON strings, bytes, and common wrapper keys
    like 'response', 'result', 'output', 'answer' which may contain
    nested stringified JSON.
    """
    parsed = None
    try:
        parsed = robust_json_loads(response)
    except Exception:
        # robust_json_loads will log; fall back to raw string
        parsed = None

    # If we got a non-dict (e.g., string), try to recover a dict
    if not isinstance(parsed, dict):
        # try to coerce via robust_json_loads again (it may raise)
        try:
            parsed = robust_json_loads(response)
        except Exception:
            # give up and wrap raw text
            return {'__raw_response__': str(response or '')}

    # At this point parsed is a dict
    for wrapper_key in ('response', 'result', 'output', 'answer'):
        if wrapper_key in parsed and parsed.get(wrapper_key):
            # attempt to parse nested
            try:
                nested = robust_json_loads(parsed.get(wrapper_key))
                if isinstance(nested, dict):
                    if '__raw_response__' not in nested:
                        nested['__raw_response__'] = parsed.get(wrapper_key)
                    return nested
            except Exception:
                # leave parsed as-is
                pass

    if '__raw_response__' not in parsed:
        parsed['__raw_response__'] = response
    return parsed


def _target_agent_id_from_key(key, anonymized_id_mapping):
    """Parse anonymized label like 'Agent 3' and map to actual agent id.

    Returns None if no numeric id found.
    """
    match = re.search(r'\d+', str(key))
    if not match:
        return None
    agent_num = int(match.group())
    # When anonymity is off the prompt uses real agent ids directly, so an
    # empty mapping should not cause us to drop valid allocations.
    if not anonymized_id_mapping:
        return agent_num
    return anonymized_id_mapping.get(agent_num)


def _parse_int_safe(val):
    try:
        return abs(int(val))
    except Exception:
        try:
            return abs(int(float(val)))
        except Exception:
            return None


def _parse_allocation_tokens(val):
    """Parse a non-negative token count; negative values mean zero (never abs())."""
    try:
        parsed = int(val)
    except Exception:
        try:
            parsed = int(float(val))
        except Exception:
            return None
    return parsed if parsed > 0 else 0


def _apply_stage2_allocations(punishments_map, rewards_map, agent, anonymized_id_mapping, group_state, budget, max_per_target):
    """Apply punishments and rewards from one shared Stage 2 budget without order bias."""
    members = group_state.get('members', []) or []
    if members:
        group_avg = sum(getattr(m, 'contribution', 0) for m in members) / len(members)
    else:
        group_avg = group_state.get('si_avg_contribution', parameters.ENDOWMENT_STAGE_1)

    punishment_requests = {}
    reward_requests = {}

    for key, raw_tokens in list((punishments_map or {}).items()):
        target_agent_id = _target_agent_id_from_key(key, anonymized_id_mapping)
        if target_agent_id is None or target_agent_id == agent.agent_id:
            continue

        tokens = _parse_allocation_tokens(raw_tokens)
        if not tokens:
            continue

        tokens = min(tokens, max_per_target)

        target_contrib = None
        for peer in members:
            if getattr(peer, 'agent_id', None) == target_agent_id:
                target_contrib = getattr(peer, 'contribution', None)
                break

        rol_enabled = bool(getattr(parameters, 'RULE_OF_LAW_ENABLED', False))
        if rol_enabled and target_contrib is not None and target_contrib >= group_avg:
            logger.warning(
                f"RULE OF LAW BLOCKED: Agent {agent.agent_id} tried to punish Agent {target_agent_id}. "
                f"Target gave {target_contrib} (Group Avg {group_avg:.2f}). Hallucinated Free-riding."
            )
            if hasattr(agent, 'rule_of_law_blocks'):
                agent.rule_of_law_blocks += 1
            continue

        punishment_requests[target_agent_id] = tokens

    for key, raw_tokens in list((rewards_map or {}).items()):
        target_agent_id = _target_agent_id_from_key(key, anonymized_id_mapping)
        if target_agent_id is None or target_agent_id == agent.agent_id:
            continue

        tokens = _parse_allocation_tokens(raw_tokens)
        if not tokens:
            continue

        tokens = min(tokens, max_per_target)
        reward_requests[target_agent_id] = tokens

    def _total_cost(pun_dict, rew_dict):
        return (
            sum(pun_dict.values()) * parameters.PUNISHMENT_COST
            + sum(rew_dict.values()) * parameters.REWARD_COST
        )

    total_cost = _total_cost(punishment_requests, reward_requests)
    if total_cost > budget and total_cost > 0:
        scale = budget / total_cost
        scaled_punishments = {}
        scaled_rewards = {}
        for target_id, tokens in punishment_requests.items():
            scaled = int(tokens * scale)
            if scaled > 0:
                scaled_punishments[target_id] = scaled
        for target_id, tokens in reward_requests.items():
            scaled = int(tokens * scale)
            if scaled > 0:
                scaled_rewards[target_id] = scaled

        while _total_cost(scaled_punishments, scaled_rewards) > budget:
            best_key = None
            best_kind = None
            best_cost = -1
            for target_id, tokens in scaled_punishments.items():
                cost = tokens * parameters.PUNISHMENT_COST
                if cost > best_cost:
                    best_cost = cost
                    best_key = target_id
                    best_kind = 'punishment'
            for target_id, tokens in scaled_rewards.items():
                cost = tokens * parameters.REWARD_COST
                if cost > best_cost:
                    best_cost = cost
                    best_key = target_id
                    best_kind = 'reward'
            if best_key is None:
                break
            if best_kind == 'punishment':
                scaled_punishments[best_key] -= 1
                if scaled_punishments[best_key] <= 0:
                    del scaled_punishments[best_key]
            else:
                scaled_rewards[best_key] -= 1
                if scaled_rewards[best_key] <= 0:
                    del scaled_rewards[best_key]

        punishment_requests = scaled_punishments
        reward_requests = scaled_rewards

    return punishment_requests, reward_requests


def _apply_allocations_helper(source_map, destination_map, tokens_remaining, cost_per_token, is_punishment, agent, anonymized_id_mapping, group_state, max_per_target):
    """Apply allocations from a source mapping (e.g., punishments or rewards) to destination_map.

    Returns (destination_map, tokens_remaining). This helper centralizes validation,
    affordability, max-per-target, Rule-of-Law blocking, and anonymized id resolution.
    """
    # Calculate group average for rule-of-law validation
    members = group_state.get('members', [])
    if members:
        group_avg = sum(getattr(m, 'contribution', 0) for m in members) / len(members)
    else:
        group_avg = group_state.get('si_avg_contribution', parameters.ENDOWMENT_STAGE_1)

    for key, raw_tokens in list(source_map.items()):
        target_agent_id = _target_agent_id_from_key(key, anonymized_id_mapping)
        if target_agent_id is None or target_agent_id == agent.agent_id:
            continue

        tokens = _parse_allocation_tokens(raw_tokens)
        if not tokens:
            continue

        tokens = min(tokens, max_per_target)
        max_affordable = tokens_remaining // cost_per_token if cost_per_token > 0 else tokens
        tokens = min(tokens, max_affordable)
        if tokens <= 0:
            continue

        if is_punishment:
            target_contrib = None
            for peer in group_state.get('members', []):
                if getattr(peer, 'agent_id', None) == target_agent_id:
                    target_contrib = getattr(peer, 'contribution', None)
                    break
            rol_enabled = bool(getattr(parameters, 'RULE_OF_LAW_ENABLED', False))
            if rol_enabled and target_contrib is not None and target_contrib >= group_avg:
                logger.warning(
                    f"RULE OF LAW BLOCKED: Agent {agent.agent_id} tried to punish Agent {target_agent_id}. "
                    f"Target gave {target_contrib} (Group Avg {group_avg:.2f}). Hallucinated Free-riding."
                )
                if hasattr(agent, 'rule_of_law_blocks'):
                    agent.rule_of_law_blocks += 1
                continue

        destination_map[target_agent_id] = tokens
        tokens_remaining -= (tokens * cost_per_token)
        if tokens_remaining <= 0:
            break

    return destination_map, tokens_remaining


def _detect_raw_shape(parsed):
    if isinstance(parsed, dict):
        return 'object'
    if isinstance(parsed, list):
        return 'list'
    if isinstance(parsed, str):
        return 'string'
    if parsed is None:
        return 'null'
    return type(parsed).__name__


def _make_parser_meta(parsed, expected_keys, fallback_used=False, fallback_reason=''):
    parsed_keys = list(parsed.keys()) if isinstance(parsed, dict) else []
    expected = set(expected_keys)
    return {
        'raw_shape': _detect_raw_shape(parsed),
        'unmapped_keys': [k for k in parsed_keys if k not in expected],
        'fallback_used': bool(fallback_used),
        'fallback_reason': str(fallback_reason or ''),
    }


def deanonymize_reasoning(reasoning, anonymized_id_mapping):
    """
    Replace anonymized agent numbers in the reasoning with actual agent IDs using a regex.
    """
    if not anonymized_id_mapping or not reasoning:
        return reasoning 

    def replacement(match):
        agent_num = int(match.group(1))
        if agent_num in anonymized_id_mapping:
            return f"Agent_ID_{anonymized_id_mapping[agent_num]}"
        return match.group(0)  # Return original if not in mapping

    return re.sub(r"Agent (\d+)", replacement, reasoning)
