# loss_damage_fund.py

from core import parameters


class LossDamageFund:
    """Persistent Loss & Damage Fund for climate-shock compensation."""

    def __init__(self):
        self.pool_balance = 0.0
        self.round_history = []

    def _safe_div(self, num, den):
        return num / den if den else 0.0

    def should_collect_contributions(self, round_number):
        """
        Determine whether this round is a replenishment round.
        """
        if getattr(parameters, 'LDF_COLLECT_EVERY_ROUND', False):
            return True

        interval = max(1, int(getattr(parameters, 'LDF_REPLENISHMENT_INTERVAL', 1)))
        return (int(round_number) % interval) == 1

    def _contribution_amount(self, agent):
        """
        Voluntary contribution amount.

        The agent chooses a round contribution elsewhere in the simulation.
        """
        return max(0.0, float(getattr(agent, 'contribution', 0.0)))

    def collect_contributions(self, agents):
        """Collect one-round contributions and return mapping {agent_id: amount}."""
        contributions = {}
        for agent in agents:
            amount = self._contribution_amount(agent)
            agent.ldf_contribution_round = amount
            contributions[agent.agent_id] = amount
            self.pool_balance += amount
        return contributions

    def distribute_payouts(self, agents, damages):
        """
        Distribute payouts to developing agents only, weighted by damage.
        Returns mapping {agent_id: payout}.
        """
        if self.pool_balance <= 0.0:
            return {}

        eligible_agents = [agent for agent in agents if getattr(agent, 'agent_group', 'developing') == 'developing']

        developing_wealths = [max(1.0, a.wealth) for a in eligible_agents]
        avg_developing_wealth = sum(developing_wealths) / len(developing_wealths) if eligible_agents else 1.0

        # Determine if pool can fully cover all eligible damages at max coverage
        max_for_agents = {}
        total_max_payout = 0.0
        for agent in eligible_agents:
            aid = agent.agent_id
            max_for_agents[aid] = damages.get(aid, 0.0) * max(0.0, parameters.LDF_MAX_COVERAGE)
            total_max_payout += max_for_agents[aid]

        # If pool balance is sufficient, set equity multiplier to 1 (no scaling)
        if self.pool_balance >= total_max_payout:
            equity_multiplier_global = 1.0
        else:
            equity_multiplier_global = None  # will be computed per agent below

        # Compute weighted need per agent
        weighted_need = {}
        for agent in eligible_agents:
            aid = agent.agent_id
            dmg = max(0.0, damages.get(aid, 0.0))
            if equity_multiplier_global is not None:
                equity_multiplier = equity_multiplier_global
            else:
                wealth_ratio = avg_developing_wealth / max(1.0, agent.wealth)
                equity_multiplier = max(0.0, 1.0 + (wealth_ratio - 1.0) * getattr(parameters, 'LDF_EQUITY_WEIGHT', 0.0))
            need = parameters.LDF_PAYOUT_DAMAGE_WEIGHT * dmg * equity_multiplier
            weighted_need[aid] = max(0.0, need)
        total_need = sum(weighted_need.values())
        if total_need <= 0.0:
            return {}

        payouts = {}
        total_pool = self.pool_balance
        remaining = self.pool_balance

        for agent in eligible_agents:
            aid = agent.agent_id
            share = self._safe_div(weighted_need[aid], total_need)
            target = total_pool * share
            max_for_agent = max_for_agents[aid]
            payout = max(0.0, min(target, max_for_agent, remaining))
            payouts[aid] = payout
            remaining -= payout

        paid_total = sum(payouts.values())
        self.pool_balance = max(0.0, self.pool_balance - paid_total)

        for agent in agents:
            payout = payouts.get(agent.agent_id, 0.0)
            agent.ldf_payout_round = payout

        return payouts

    def append_round_log(self, log_row):
        self.round_history.append(log_row)
