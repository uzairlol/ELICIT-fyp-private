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


def _format_recent_institutions(agent, limit=3):
    recent = getattr(agent, 'history_institutions', [])[-limit:]
    return ", ".join(recent) if recent else "none"
