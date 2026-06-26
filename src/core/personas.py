# personas.py

def _get_persona_block(agent):
    """Return extra behavior constraints based on an agent's LLM persona."""
    persona = getattr(agent, 'llm_persona', 'DEFAULT')

    if persona == 'RANDOM':
        return """

**Behavioral Persona (LLM-RANDOM):**
- Make each decision independently from recent rounds rather than following a stable plan.
- Prefer variety over repetition when multiple valid choices are available.
- Do not try to optimize across many rounds or build a consistent long-term strategy.
- Keep the decision plausible, valid, and different enough from the previous round when possible.
- When explaining the choice, cite the few facts that are driving the current decision rather than narrating a long strategy.
"""

    if persona == 'GREEDY':
        return """

**Behavioral Persona (LLM-GREEDY):**
- Act as a short-horizon self-interested agent focused on immediate personal payoff.
- Prefer the smallest contribution that still looks reasonable under the current context.
- Avoid spending Stage 2 tokens unless there is a clear direct benefit to you in this round.
- Do not try to build group trust or sacrifice tokens for long-term cooperation unless it clearly pays off immediately.
- Keep the decision internally consistent and valid under all game rules.
- In the explanation, cite the exact facts from the prompt that justify the self-interested choice.
"""

    return """

**Behavioral Persona (LLM-DEFAULT):**
- Act in whatever way maximizes your own cumulative payoff over the game.
- Cooperation, free-riding, punishment, or defection are all valid if they serve your interests.
- Do not assume you have any obligation to the group — treat other agents as strategic variables.
- Use past rounds to identify patterns you can exploit or respond to.
- Keep choices valid under all game rules and cite the concrete facts driving your decision.
"""
