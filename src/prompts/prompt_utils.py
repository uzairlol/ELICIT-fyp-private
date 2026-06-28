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


def _agent_task_header(agent, task_label, round_number, extra=""):
    """Short role + task line for small models (repeat task at top of prompt)."""
    suffix = f" {extra}" if extra else ""
    return (
        f"You are Agent {agent.agent_id}. Round {round_number}. "
        f"Task: {task_label}.{suffix}\n"
    )


_LLM_FINAL_OUTPUT_RULES = """
**FINAL OUTPUT RULES (read last — mandatory for all responses):**
- Do all thinking internally. Do NOT write analysis, markdown, or commentary outside the JSON.
- Return exactly ONE JSON object. No code fences. No text before or after the JSON.
- Use the exact key names shown in the Required JSON shape below.
"""


def _llm_decision_steps(stage_name):
    """Numbered steps help 8B models separate reasoning from structured output."""
    steps = {
        "Institution Choice": (
            "1. Read the decision card, beliefs, and peer data.\n"
            "2. Choose SI or SFI based on your payoff interests.\n"
            "3. Fill institution_choice, reasoning, and facts_used in the JSON below."
        ),
        "Contribution Choice": (
            "1. Read the Stage 1 decision card and MCPR (if shown).\n"
            "2. Pick one integer contribution within the allowed budget.\n"
            "3. Fill contribution, reasoning, and facts_used in the JSON below."
        ),
        "Punishment and Reward Choice": (
            "1. Review each target's contribution and stated intent.\n"
            "2. Set integer amounts in the \"punishments\" object (0 if not punishing).\n"
            "3. Fill \"justifications\" for every target label.\n"
            "4. Write a one-sentence \"reasoning\" summary (amounts stay in punishments).\n"
            "5. Output the JSON object below — nothing else."
        ),
    }
    body = steps.get(stage_name, "Follow the response contract below.")
    return f"\n**Decision steps:**\n{body}\n"

