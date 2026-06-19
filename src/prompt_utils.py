#prompt_utils.py

def _safe_int(value, default=0):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_token_list(values):
    values = list(values or [])
    if not values:
        return "none"
    return ", ".join(str(_safe_int(v)) for v in values)


def _format_recent_rounds(agent, limit=3):
    recent_rounds = getattr(agent, 'history', [])[-limit:]
    if not recent_rounds:
        return "None yet."

    lines = []
    for entry in recent_rounds:
        round_number = entry.get('round_number', '?')
        institution = entry.get('institution_choice', 'unknown')
        contribution = _safe_int(entry.get('contribution', 0))
        stage1_payoff = _safe_float(entry.get('stage1_payoff', 0.0))
        stage2_payoff = _safe_float(entry.get('stage2_payoff', 0.0))
        total_payoff = _safe_float(entry.get('payoff', 0.0))
        lines.append(
            f"R{round_number}: inst={institution}, contrib={contribution}, "
            f"S1={stage1_payoff:.2f}, S2={stage2_payoff:.2f}, total={total_payoff:.2f}"
        )
    return "\n".join(lines)


def _format_recent_institutions(agent, limit=3):
    recent = getattr(agent, 'history_institutions', [])[-limit:]
    return ", ".join(recent) if recent else "none"
