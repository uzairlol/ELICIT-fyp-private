# oracle.py


# version 1


"""
Oracle Module.

The Oracle is a "Synthetic Heuristic Oracle", representing an environmental,
rule-based welfare predictor used during democratic votes. It evaluates each
proposed rule change and estimates its effect on group social welfare based on:

  - Current observed cooperation rate (avg contribution / max contribution)
  - Current free-rider proportion (agents below 50% of average)
  - Direction and magnitude of the proposed change

This gives agents data-grounded context when voting, replacing pure LLM
speculation with calculated signal. Note for research applications: this is 
a synthetic environmental heuristic rather than an objective truth model or 
trained critic.

Usage:
    oracle = Oracle(round_history, agents, parameters)
    annotations = oracle.annotate_proposals(proposals)
    # Returns list of strings like:
    # "PUNISHMENT_EFFECT 3→4: Oracle predicts +14% welfare. Free-rider rate: 33%."
"""

from core import parameters


class Oracle:
    """
    Computes welfare predictions for democracy proposals.
    """

    def __init__(self, round_history: list, agents: list):
        """
        Args:
            round_history: list of round result dicts from environment.results
            agents: list of Agent instances
        """
        self.round_history = round_history
        self.agents = agents

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_proposals(self, proposals: list) -> list:
        """
        For each proposal dict, compute a welfare annotation string.

        Returns:
            list of str — one annotation per proposal, in order.
        """
        stats = self._compute_current_stats()
        annotations = []
        for p in proposals:
            ann = self._annotate_single(p, stats)
            annotations.append(ann)
        return annotations

    # ------------------------------------------------------------------
    # Statistics from recent history
    # ------------------------------------------------------------------

    def _compute_current_stats(self) -> dict:
        """
        Aggregate stats from the last 3 rounds (or all rounds if fewer).
        """
        recent = self.round_history[-3:] if len(self.round_history) >= 3 else self.round_history

        all_contribs = []
        for rnd in recent:
            for agent_data in rnd.get('agents', {}).values():
                all_contribs.append(agent_data.get('contribution', 0))

        n = len(all_contribs)
        avg_contrib = sum(all_contribs) / n if n > 0 else 0
        max_c = parameters.MAX_CONTRIBUTION

        # Cooperation rate: avg contribution as fraction of max
        coop_rate = avg_contrib / max_c if max_c > 0 else 0

        # Free-rider rate: fraction of contributions below half the average
        threshold = avg_contrib * 0.5
        freerider_count = sum(1 for c in all_contribs if c < threshold)
        freerider_rate = freerider_count / n if n > 0 else 0

        # Recent avg cumulative payoff trend
        payoffs = []
        for rnd in recent:
            for agent_data in rnd.get('agents', {}).values():
                payoffs.append(agent_data.get('payoff', 0))
        avg_payoff = sum(payoffs) / len(payoffs) if payoffs else 0

        return {
            'avg_contrib': round(avg_contrib, 1),
            'coop_rate': round(coop_rate, 3),
            'freerider_rate': round(freerider_rate, 3),
            'avg_payoff': round(avg_payoff, 1),
            'n_observations': n,
        }

    # ------------------------------------------------------------------
    # Per-proposal annotation
    # ------------------------------------------------------------------

    def _annotate_single(self, proposal: dict, stats: dict) -> str:
        """
        Compute a welfare annotation string for one proposal.
        """
        rule = proposal.get('rule', '')
        new_val = proposal.get('new_value', None)
        old_val = getattr(parameters, rule, None)

        if old_val is None or new_val is None:
            return f"{rule}: Oracle has insufficient data to predict welfare impact."

        if old_val == 0:
            change_pct = float(new_val)
        else:
            change_pct = (new_val - old_val) / old_val
        fr = stats['freerider_rate']
        coop = stats['coop_rate']

        if rule == 'PUNISHMENT_EFFECT':
            # Higher punishment helps when free-rider rate is high
            # Predicted gain = change_pct * freerider_rate * base_gain_factor
            predicted_welfare_pct = change_pct * fr * 100 * parameters.ORACLE_PUNISHMENT_WEIGHT
            direction = "increase deterrence" if change_pct > 0 else "reduce punishment burden"
            rationale = (
                f"Free-rider rate: {fr*100:.0f}%. "
                f"Harsher punishment helps more when free-riding is high. "
                f"Predicted {'welfare gain' if predicted_welfare_pct > 0 else 'welfare cost'}: "
                f"{predicted_welfare_pct:+.1f}%."
            )

        elif rule == 'REWARD_EFFECT':
            # Rewards help when cooperation rate is already moderate (encourages more)
            predicted_welfare_pct = change_pct * coop * 100 * parameters.ORACLE_REWARD_WEIGHT
            rationale = (
                f"Cooperation rate: {coop*100:.0f}%. "
                f"Rewards amplify already-cooperative behaviour. "
                f"Predicted {'welfare gain' if predicted_welfare_pct > 0 else 'welfare cost'}: "
                f"{predicted_welfare_pct:+.1f}%."
            )

        elif rule == 'ENDOWMENT_STAGE_2':
            # More budget = more punishment/reward capacity but also more redistribution cost
            predicted_welfare_pct = change_pct * parameters.ORACLE_ENDOWMENT_SCALING  # Moderate flat scaling
            rationale = (
                f"Current avg round payoff: {stats['avg_payoff']:.1f} tokens. "
                f"Larger Stage 2 budget gives more sanctioning capacity. "
                f"Predicted redistribution effect: {predicted_welfare_pct:+.1f}%."
            )

        elif rule == 'MAX_PUNISHMENT_TOKENS':
            # Cap on assignable tokens — mainly affects punishment intensity ceiling
            predicted_welfare_pct = change_pct * fr * parameters.ORACLE_MAX_TOKENS_WEIGHT
            rationale = (
                f"Free-rider rate: {fr*100:.0f}%. "
                f"Raising the cap matters most when free-riding is high. "
                f"Predicted effect on welfare: {predicted_welfare_pct:+.1f}%."
            )

        elif rule in {
            'LDF_PAYOUT_DAMAGE_WEIGHT',
            'LDF_MAX_COVERAGE',
            'LDF_EQUITY_WEIGHT',
        }:
            # Heuristic: LDF expansion helps most when cooperation is moderate+ and
            # free-riding is not extreme, because redistribution credibility improves resilience.
            resilience_signal = max(0.0, (coop - (fr * 0.5)))
            predicted_welfare_pct = change_pct * 100 * (8.0 + 20.0 * resilience_signal)
            rationale = (
                f"Cooperation rate: {coop*100:.0f}%, free-rider rate: {fr*100:.0f}%. "
                f"This rule tunes climate-loss redistribution capacity. "
                f"Predicted resilience impact: {predicted_welfare_pct:+.1f}%."
            )

        else:
            return f"{rule} {old_val}→{new_val}: Oracle has no model for this rule."

        return (
            f"{rule} {old_val}→{new_val}: {rationale}"
        )
