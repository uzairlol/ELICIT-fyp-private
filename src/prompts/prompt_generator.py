# prompt_generator.py

from core import parameters
from core.scenario_config import get_scenario_config
from prompts.prompt_utils import _safe_int, _safe_float, _format_token_list, _format_recent_institutions
from core.personas import _get_persona_block
from core.utils import uses_climate_budget


def _uses_climate_budget():
    return uses_climate_budget()


def _contribution_budget(agent):
    if _uses_climate_budget():
        return max(parameters.MIN_CONTRIBUTION, int(getattr(agent, 'wealth', parameters.ENDOWMENT_STAGE_1)))
    return parameters.ENDOWMENT_STAGE_1


def _should_broadcast_country_types():
    scenario_name = str(getattr(parameters, 'SCENARIO', '')).lower()
    if scenario_name == 'climate':
        scenario_name = 'ldf'

    climate_mode = (
        scenario_name == 'ldf'
        or bool(getattr(parameters, 'CLIMATE_SHOCK_ENABLED', False))
        or bool(getattr(parameters, 'LDF_ENABLED', False))
    )
    if not climate_mode:
        return False

    batch_name = str(getattr(parameters, 'BATCH_NAME', '')).lower()
    if batch_name.startswith('control'):
        return False

    return True


def _use_anonymity():
    """Use anonymity outside climate/LDF; disable it in climate/LDF mode."""
    scenario_name = str(getattr(parameters, 'SCENARIO', '')).lower()
    if scenario_name == 'climate':
        scenario_name = 'ldf'

    climate_mode = (
        scenario_name == 'ldf'
        or bool(getattr(parameters, 'CLIMATE_SHOCK_ENABLED', False))
        or bool(getattr(parameters, 'LDF_ENABLED', False))
    )
    if climate_mode:
        return False

    return bool(getattr(parameters, 'ANONYMITY', False))


def _peer_label(observer_agent, actual_agent_id, fallback_number):
    if _use_anonymity() and hasattr(observer_agent, 'pseudonym_mapping'):
        return str(observer_agent.pseudonym_mapping.get(actual_agent_id, 'X'))
    return str(actual_agent_id)


def _received_tokens_from_effect(effect, effect_per_token):
    """Convert stored payoff effect back to token count for prompt display."""
    if effect_per_token <= 0:
        return 0
    return int(round(float(effect or 0) / effect_per_token))


def _sort_peers_for_punishment(others):
    """List peers worst free-rider first (lowest contribution), then by agent_id."""
    return sorted(
        others,
        key=lambda member: (_safe_int(getattr(member, 'contribution', 0)), member.agent_id),
    )

def _json_response_block(stage_name):
    return f"""

**Response Contract ({stage_name}):**
- Think briefly before answering, but keep the final response concise.
- Use at most 2-3 short internal reasoning steps.
- Return exactly one JSON object and nothing else.
- Keep `reasoning` to one short sentence.
- Keep `facts_used` to the 2-3 most important facts only.
- Do not add markdown, code fences, or commentary outside the JSON.
"""





def _build_common_snapshot(agent, round_number, sc, stage_text):
    cumulative_payoff = _safe_float(getattr(agent, 'cumulative_payoff', parameters.INITIAL_TOKENS))
    agent_group = getattr(agent, 'agent_group', 'developing')
    vulnerability = _safe_float(getattr(agent, 'vulnerability', 1.0))
    emissions = _safe_float(getattr(agent, 'historical_emissions', 0.0))
    wealth = _safe_float(getattr(agent, 'wealth', cumulative_payoff))
    ldf_contrib = _safe_float(getattr(agent, 'ldf_contribution_round', 0.0))
    ldf_payout = _safe_float(getattr(agent, 'ldf_payout_round', 0.0))
    climate_damage = _safe_float(getattr(agent, 'climate_damage_taken_round', 0.0))

    return f"""
You are participating in a {sc['game_name']} in Round {round_number}.
{stage_text}

**Decision Snapshot:**
- Currency: {sc['currency_name']}
- Account: {sc['account_name']}
- Project: {sc['project_name']}
- Current wealth: {wealth:.2f} {sc['currency_name']}
- Cumulative payoff: {cumulative_payoff:.2f} {sc['currency_name']}
- Group type: {agent_group}
- Vulnerability index: {vulnerability:.2f}
- Historical emissions index: {emissions:.2f}
- LDF contribution this round: {ldf_contrib:.2f} {sc['currency_name']}
- LDF payout this round: {ldf_payout:.2f} {sc['currency_name']}
- Climate damage this round: {climate_damage:.2f} {sc['currency_name']}
"""


def _build_stage0_card(agent, round_number, sc):
    current_budget = _safe_int(getattr(agent, 'wealth', parameters.INITIAL_TOKENS))
    recent_contributions = _format_token_list(getattr(agent, 'history_contributions', [])[-3:])
    recent_institutions = _format_recent_institutions(agent)
    
    s2_budget = agent.get_stage2_budget() if hasattr(agent, 'get_stage2_budget') else parameters.ENDOWMENT_STAGE_2
    
    p_cost = parameters.PUNISHMENT_COST
    p_effect = parameters.PUNISHMENT_EFFECT
    r_cost = parameters.REWARD_COST
    r_effect = parameters.REWARD_EFFECT
    
    if _uses_climate_budget():
        si_endowment_desc = (
            f"You do NOT receive a free Stage 2 endowment. Instead, you can spend up to your Stage 2 budget "
            f"({s2_budget} {sc['currency_name']}, calculated as 5% of your wealth, or at least {parameters.ENDOWMENT_STAGE_2} {sc['currency_name']}) "
            f"directly from your OWN wealth."
        )
        si_payoff_formula = f"`stage1_payoff - amount_you_spend + rewards_you_receive - punishments_you_receive` (costs are deducted directly from your wealth)"
        key_tradeoff_desc = (
            f"The risk is that peers in SI may punish you if they think you under-contributed. Any {sc['currency_name']} you spend "
            f"or receive as punishment directly reduce your wealth, while rewards increase it."
        )
        unspent_explanation = "In climate mode, you simply spend 0 from your wealth."
    else:
        si_endowment_desc = f"You receive a FREE Stage 2 endowment of {s2_budget} {sc['currency_name']} just for joining SI (SFI members get nothing from Stage 2)."
        si_payoff_formula = f"`stage1_payoff + ({s2_budget} - amount_you_spend) + rewards_you_receive - punishments_you_receive`"
        key_tradeoff_desc = (
            f"SI gives you {s2_budget} {sc['currency_name']} extra per round that SFI does not. The risk is that peers in SI "
            f"may punish you if they think you under-contributed, reducing your net Stage 2 payoff."
        )
        unspent_explanation = f"If you spend nothing in Stage 2, you automatically pocket the full {s2_budget} {sc['currency_name']} endowment on top of your Stage 1 payoff."

    return f"""
**Decision Card — Stage 0 / Institution Choice**

You must choose between two institutions:

- **SI ({sc['si_name']})**: {sc['si_desc']}.
  - {si_endowment_desc}
  - In Stage 2 you may spend some or all of that budget to punish (cost {p_cost}, effect -{p_effect}) or reward (cost {r_cost}, effect +{r_effect}) other SI members.
  - You can also be punished or rewarded by others in your SI group.
  - **SI payoff formula:** {si_payoff_formula}
  - {unspent_explanation}

- **SFI ({sc['sfi_name']})**: {sc['sfi_desc']}.
  - No Stage 2 budget, no Stage 2 spending, no punishment/reward.
  - **SFI payoff formula:** `stage1_payoff` only.

**Key trade-off:** {key_tradeoff_desc}

- Current budget marker: {current_budget} {sc['currency_name']}
- Recent institution choices: {recent_institutions}
- Recent contributions: {recent_contributions}
- Peer trust score: {getattr(agent, 'reputation', 5.0):.1f}/10

Choose the institution that best fits your self-interest in this round.
"""


def _build_stage1_card(agent, group_state, sc):
    budget = _safe_int(_contribution_budget(agent))
    recent_contributions = _format_token_list(getattr(agent, 'history_contributions', [])[-3:])
    recent_institutions = _format_recent_institutions(agent)
    group_members = list((group_state or {}).get('members', []) or [])
    group_size = len(group_members)

    prev_avg_str = "N/A (first round)"
    if getattr(agent, 'anonymous_data_history', None):
        latest = agent.anonymous_data_history[-1]
        same_inst_contribs = [
            entry['contribution'] for entry in latest.get('anonymous_data', [])
            if entry.get('institution_choice') == getattr(agent, 'institution_choice', None)
        ]
        if same_inst_contribs:
            prev_avg_str = f"{sum(same_inst_contribs) / len(same_inst_contribs):.2f} {sc['currency_name']}"

    return f"""
**Decision Card — Stage 1 / Contribution**
- Institution: {getattr(agent, 'institution_choice', 'unknown')}
- Contribution budget: {budget} {sc['currency_name']}
- Minimum contribution: {parameters.MIN_CONTRIBUTION}
- Maximum contribution: {budget}
- Group size: {group_size}
- Previous round's group average contribution: {prev_avg_str}
- Your recent contributions: {recent_contributions}
- Your recent institution choices: {recent_institutions}

Choose one integer contribution within the allowed budget.
"""


def _build_stage2_card(agent, group_state, sc, ordered_others=None):
    """Build the Stage 2 card. If `ordered_others` is provided, use that ordering deterministically."""
    members = list((group_state or {}).get('members', []) or [])
    stage1_payoffs = dict((group_state or {}).get('stage1_payoffs', {}) or {})
    use_anonymity = _use_anonymity()
    show_country_types = _should_broadcast_country_types()

    if ordered_others is None:
        others = [member for member in members if member.agent_id != agent.agent_id]
        others = _sort_peers_for_punishment(others)
    else:
        others = ordered_others

    total_contrib = sum(_safe_int(getattr(member, 'contribution', 0)) for member in members)
    avg_contrib = (total_contrib / len(members)) if members else 0.0
    tom_scores = getattr(agent, 'tom_scores', {}) or {}

    rows = []
    target_labels = []

    for member in others:
        label_id = member.agent_id if not use_anonymity else agent.pseudonym_mapping.get(member.agent_id, -1)
        if use_anonymity and label_id == -1:
            continue

        label = f"Agent {label_id}"
        target_labels.append(label)
        agent.anonymized_id_mapping[label_id] = member.agent_id

        stage1_budget = member.get_stage1_contribution_cap() if hasattr(member, 'get_stage1_contribution_cap') else parameters.ENDOWMENT_STAGE_1
        contribution = _safe_int(getattr(member, 'contribution', 0))
        kept = max(0, _safe_int(stage1_budget) - contribution)
        stage1_payoff = _safe_float(stage1_payoffs.get(member.agent_id, 0.0))
        group_type = f", group={getattr(member, 'agent_group', 'unknown')}" if show_country_types else ""

        deviation = contribution - avg_contrib
        free_rider_tag = " [FREE-RIDER — contributed BELOW average]" if contribution < avg_contrib else ""
        stated_reason = (getattr(member, 'contribution_reasoning', '') or 'No reasoning provided.').strip()
        if len(stated_reason) > 180:
            stated_reason = stated_reason[:180] + '...'

        tom_suffix = ""
        if parameters.TOM_ENABLED:
            peer_score = tom_scores.get(member.agent_id)
            if peer_score is not None:
                tom_suffix = f", your trust score for them: {float(peer_score):.1f}/10"

        rows.append(
            f"- {label}{group_type}{free_rider_tag}{tom_suffix}: "
            f"budget/wealth={float(stage1_budget):.2f} {sc['currency_name']}, "
            f"contrib={contribution} {sc['currency_name']} (deviation from avg: {deviation:+.1f}), "
            f"kept={kept} {sc['currency_name']}, "
            f"net_stage1_payoff={stage1_payoff:.2f} {sc['currency_name']}, "
            f"stated intent: \"{stated_reason}\""
        )

    targets_block = "\n".join(rows) if rows else "- No other targets available."
    target_label_block = ", ".join(target_labels) if target_labels else "none"

    s2_budget = agent.get_stage2_budget() if hasattr(agent, 'get_stage2_budget') else parameters.ENDOWMENT_STAGE_2
    max_punishment = agent.get_max_punishment_tokens() if hasattr(agent, 'get_max_punishment_tokens') else parameters.MAX_PUNISHMENT_TOKENS
    funding_info = " (funded directly from your wealth)" if _uses_climate_budget() else ""

    card = f"""
**Decision Card — Stage 2 / Punishment & Reward Allocation**
- Your institution: {getattr(agent, 'institution_choice', 'unknown')}
- Stage 2 budget: {s2_budget} {sc['currency_name']} TOTAL across all targets combined (punishments + rewards share this pool){funding_info}
- Max amount assignable to any single target: {max_punishment} {sc['currency_name']}
- Punishment effect / cost: -{parameters.PUNISHMENT_EFFECT} / {parameters.PUNISHMENT_COST} {sc['currency_name']} per unit
- Reward effect / cost: +{parameters.REWARD_EFFECT} / {parameters.REWARD_COST} {sc['currency_name']} per unit
- Group average contribution this round: {avg_contrib:.2f} {sc['currency_name']} — agents below this are free-riding
- Target labels (use exactly): {target_label_block}
- Targets are listed worst free-rider first — prioritize punishing agents with the lowest contributions and mismatched stated intent.

Current-round targets (agents marked [FREE-RIDER] contributed below the group average):
{targets_block}

Allocate integer amounts only. Include every listed target exactly once in "punishments" (use 0 if not punishing). Rewards are optional.
"""

    return card, target_labels

def get_past_actions_string(agent):
    """
    Helper method to format past actions for prompts.
    Returns A formatted string of past actions and outcomes.
    """
    sc = get_scenario_config(parameters.SCENARIO)
    currency_name = sc['currency_name']
    punishment_name = sc['punishment_name']
    reward_name = sc['reward_name']

    actions = []
    for entry in agent.history[-parameters.DISPLAY_PAST_ACTIONS:]:
        ap = entry.get('assigned_punishments', {})
        ar = entry.get('assigned_rewards', {})
        assigned_p_tokens = sum(ap.values()) if isinstance(ap, dict) else ap
        assigned_r_tokens = sum(ar.values()) if isinstance(ar, dict) else ar
        received_p_tokens = _received_tokens_from_effect(
            entry.get('received_punishments', 0), parameters.PUNISHMENT_EFFECT
        )
        received_r_tokens = _received_tokens_from_effect(
            entry.get('received_rewards', 0), parameters.REWARD_EFFECT
        )
        round_info = f"Round {entry['round_number']}: " \
                     f"Institution: {entry['institution_choice']}, " \
                     f"Contribution: {entry['contribution']}, " \
                     f"Assigned {punishment_name}: {assigned_p_tokens}, " \
                     f"Assigned {reward_name}: {assigned_r_tokens}, " \
                     f"Received {punishment_name}: {received_p_tokens}, " \
                     f"Received {reward_name}: {received_r_tokens}, " \
                     f"Stage 1 Payoff: {entry.get('stage1_payoff', 0):.2f}, " \
                     f"Stage 2 Payoff: {entry.get('stage2_payoff', 0):.2f}, " \
                     f"Total Round Payoff: {entry.get('payoff', 0):.2f} {currency_name}, " \
                     f"Cumulative Payoff: {entry.get('cumulative_payoff', 0):.2f} {currency_name}"
        
        # Add Tom Reputation Score (peer trust) if ToM is enabled
        if parameters.TOM_ENABLED and 'reputation' in entry:
            round_info += f", Reputation Score: {entry['reputation']:.1f}"

        actions.append(round_info)
    
    if not actions:
        return "No past actions."
    
    return "\n".join(actions)

def __build_prompt_prefix(agent, stage_text, round_number, sc):
    """Build the compact shared prompt prefix."""
    prompt = _build_common_snapshot(agent, round_number, sc, stage_text)
    
    s2_budget = agent.get_stage2_budget() if hasattr(agent, 'get_stage2_budget') else parameters.ENDOWMENT_STAGE_2
    funding_info = " (funded directly from wealth, no free endowment)" if _uses_climate_budget() else " endowment"

    prompt += f"""

**Scenario Rules Snapshot:**
- Stage 1 contribution uses your current budget: {_safe_int(agent.get_stage1_contribution_cap() if hasattr(agent, 'get_stage1_contribution_cap') else parameters.ENDOWMENT_STAGE_1)} {sc['currency_name']}
- Public good multiplier: {parameters.PUBLIC_GOOD_MULTIPLIER}
- Stage 2 budget: {s2_budget} {sc['currency_name']}{funding_info} (SI members only — SFI members skip Stage 2 entirely)
- Punishment effect / cost (SI only): -{parameters.PUNISHMENT_EFFECT} / {parameters.PUNISHMENT_COST} {sc['currency_name']}
- Reward effect / cost (SI only): +{parameters.REWARD_EFFECT} / {parameters.REWARD_COST} {sc['currency_name']}
"""
    return prompt

def _append_belief_state(prompt, agent, sc):
    """Inject the agent's structured belief-state scratchpad and T-1 peer data."""
    import json as _json

    belief = getattr(agent, 'belief_state', None)
    if belief:
        prompt += f"\n\n**Your Internal Beliefs (Working Memory):**\n{_json.dumps(belief, indent=2)}"
    else:
        prompt += "\n\n**Your Internal Beliefs (Working Memory):** No beliefs formed yet."

    # Show only the single most-recent round of peer data (T-1)
    show_country_types = _should_broadcast_country_types()
    if agent.anonymous_data_history:
        latest = agent.anonymous_data_history[-1]
        round_num = latest['round_number']
        anonymous_data_list = latest['anonymous_data']
        heading = "Anonymous Data" if _use_anonymity() else "Peer Data"
        prompt += f"\n\n**{heading} from Previous Round (Round {round_num}):**"
        for i, entry in enumerate(anonymous_data_list):
            if _use_anonymity() and entry.get('actual_agent_id', -1) not in getattr(agent, 'pseudonym_mapping', {}):
                continue
            label = _peer_label(agent, entry.get('actual_agent_id', -1), i + 1)
            country_str = f"Country type: {entry.get('agent_group', 'unknown')}, " if show_country_types else ""
            received_p_tokens = _received_tokens_from_effect(
                entry.get('received_punishments', 0), parameters.PUNISHMENT_EFFECT
            )
            received_r_tokens = _received_tokens_from_effect(
                entry.get('received_rewards', 0), parameters.REWARD_EFFECT
            )
            prompt += (
                f"\nAgent {label}: {country_str}"
                f"Institution: {entry.get('institution_choice', 'Unknown')}, "
                f"Budget/Wealth: {entry.get('wealth', 0.0):.2f} {sc['currency_name']}, "
                f"Contributed {entry['contribution']} {sc['currency_name']}, "
                f"Received {sc['punishment_name']}: {received_p_tokens}, "
                f"Received {sc['reward_name']}: {received_r_tokens}, "
                f"Total Round Payoff: {entry.get('total_round_payoff', 0):.2f}"
            )
        prompt += "\n\nUse your Internal Beliefs above for long-term context and the previous round data for immediate situational awareness."
    else:
        prompt += "\n\nNo peer data from previous rounds is available yet."
    return prompt

def _append_gossip(prompt, agent):
    """Append prior-round gossip bulletin when available."""
    if parameters.GOSSIP_ENABLED:
        gossip_text = getattr(agent, 'recent_gossip', '')
        if gossip_text:
            prompt += "\n\nSocial Reputation Bulletin (from previous round):"
            prompt += f"\n{gossip_text}"
            prompt += "\n\nTreat this as soft social evidence and combine it with observed contributions and outcomes."
    return prompt


def _append_climate_role_guidance(prompt, agent):
    """Add scenario-specific soft guidance for climate framing."""
    if str(parameters.SCENARIO).lower() != "ldf":
        return prompt

    group = getattr(agent, 'agent_group', 'developing')
    if group == 'developed':
        prompt += """

**Climate Role Guidance (Developed Country):**
- You typically have higher fiscal capacity and historically higher emissions.
- Treat long-run climate stability and institutional credibility as strategically important.
- You may choose either self-interested or cooperative actions, but explain trade-offs clearly.
"""
    else:
        prompt += """

**Climate Role Guidance (Developing Country):**
- You typically face higher vulnerability and tighter budget constraints.
- Balance immediate resilience needs against long-run cooperation incentives.
- You may choose either protective or cooperative actions, but explain trade-offs clearly.
"""

    prompt += """

**LDF and Shock Context Reminder:**
- Climate shocks are deterministic and can create asymmetric losses.
- The Loss & Damage Fund can shift net outcomes through contributions and payouts.
- **CRITICAL:** Any {sc['currency_name']} you choose to contribute to the public good (Emissions Reduction) are ALSO deposited into the LDF pool to fund disaster payouts.
- Use these mechanisms as decision context, not as fixed rules that force a single choice.
"""
    return prompt


def construct_institution_choice_prompt(agent, round_number):
    sc = get_scenario_config(parameters.SCENARIO)
    stage_text = f"You are in the institution selection stage in Round {round_number}."
    
    prompt = __build_prompt_prefix(agent, stage_text, round_number, sc)
    prompt += _get_persona_block(agent)
    prompt = _append_climate_role_guidance(prompt, agent)
    prompt += _build_stage0_card(agent, round_number, sc)

    if parameters.CURIOSITY_ENABLED and parameters.CURIOSITY_BONUS_PROMPT:
        if len(agent.history_institutions) >= 3:
            last_3 = set(agent.history_institutions[-3:])
            if len(last_3) == 1:
                regime = list(last_3)[0]
                other = "SI" if regime == "SFI" else "SFI"
                prompt += f"\n**Note:** You have chosen {regime} for the last 3 rounds. Exploring {other} might provide additional data on how different rules affect outcomes.\n"

    prompt = _append_belief_state(prompt, agent, sc)
    prompt = _append_gossip(prompt, agent)

    if parameters.TOM_ENABLED:
        prompt += f"\n\n**Peer Trust Rating (Current Score):** {agent.reputation:.1f}/10"

    prompt += "\n\nDecide which institution to join. Use the decision card above and keep the final answer short."
    prompt += _json_response_block("Institution Choice")
    return prompt.strip()


def construct_contribution_prompt(agent, group_state):
    sc = get_scenario_config(parameters.SCENARIO)
    stage_text = f"You are in Stage 1, in the {agent.institution_choice} in Round {agent.round_number}."
    prompt = __build_prompt_prefix(agent, stage_text, agent.round_number, sc)
    prompt += _get_persona_block(agent)
    prompt = _append_climate_role_guidance(prompt, agent)
    prompt += _build_stage1_card(agent, group_state, sc)
    prompt = _append_belief_state(prompt, agent, sc)
    prompt = _append_gossip(prompt, agent)

    prompt += "\n\nDecide how much to contribute. Use the card above and return one integer value within the allowed budget. Keep reasoning concise."
    prompt += _json_response_block("Contribution Choice")

    if parameters.CURIOSITY_ENABLED and parameters.CURIOSITY_BONUS_PROMPT:
        if len(agent.history_contributions) >= 3:
            last_3 = agent.history_contributions[-3:]
            if max(last_3) - min(last_3) <= 2:
                prompt += f"\n**Note:** Your contribution has been consistent at {last_3[-1]} {sc['currency_name']}. Experimenting with different values may provide insights into group dynamics.\n"

    return prompt.strip()


def construct_punishment_prompt(agent, group_state):
    sc = get_scenario_config(parameters.SCENARIO)
    stage_text = f"You are in Stage 2, in the {agent.institution_choice} in Round {agent.round_number}."
    # Reset mapping for this prompt
    agent.anonymized_id_mapping = {}

    s2_budget = agent.get_stage2_budget() if hasattr(agent, 'get_stage2_budget') else parameters.ENDOWMENT_STAGE_2

    prompt = __build_prompt_prefix(agent, stage_text, agent.round_number, sc)
    prompt += _get_persona_block(agent)
    prompt = _append_climate_role_guidance(prompt, agent)

    members = list((group_state or {}).get('members', []) or [])
    ordered_others = _sort_peers_for_punishment(
        [m for m in members if m.agent_id != agent.agent_id]
    )
    card_text, target_labels = _build_stage2_card(agent, group_state, sc, ordered_others=ordered_others)
    prompt += card_text

    prompt = _append_belief_state(prompt, agent, sc)
    prompt = _append_gossip(prompt, agent)

    if parameters.TOM_ENABLED:
        prompt += f"\n\n**Your Peer Trust Rating (aggregate):** {agent.reputation:.1f}/10"
        prompt += "\nUse per-target trust scores in the decision card together with current contributions and stated intent."

    prompt += (
        "\n\n**Important:** Prioritize current-round contributions and stated intent (above) when deciding punishments. "
        "Use gossip, trust scores, and Internal Beliefs as secondary evidence."
    )

    label_block = ", ".join(target_labels) if target_labels else ""

    def _json_kv(labels, value):
        return ",\n        ".join(f'"{lbl}": {value}' for lbl in labels)

    punish_keys = _json_kv(target_labels, 0)

    prompt += f"""

**Response Contract (Punishment and Reward Choice):**
- Return exactly one JSON object and nothing else.
- "punishments" MUST include ALL {len(target_labels)} target labels as keys. Use 0 for targets you are not punishing.
- "rewards" is OPTIONAL — omit it or use {{}} if you are not rewarding anyone. If present, only include targets with a positive amount.
- "justifications" is OPTIONAL — if included, only justify targets you punish or reward with a positive amount.
- Use integer amounts only (never negative).
- Your total spend (punishment amount × {parameters.PUNISHMENT_COST} + reward amount × {parameters.REWARD_COST}) MUST NOT exceed {s2_budget}.
- Do not include yourself in either object.
- Focus punishments on free-riders: agents who contributed below the group average or whose stated intent does not match their action.

**Target labels (ALL must appear as keys in punishments):** {label_block}

**Required JSON shape:**
{{
    "punishments": {{
        {punish_keys}
    }},
    "reasoning": "One or two sentences explaining your overall strategy.",
    "facts_used": ["Fact 1", "Fact 2"]
}}

- Keep `reasoning` to the point.
- Keep `facts_used` to the 2-3 most important facts only.
- Do not add markdown, code fences, or commentary outside the JSON.
"""
    return prompt.strip()
