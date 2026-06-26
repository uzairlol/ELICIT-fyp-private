# subsidy.py

from core import parameters

class SubsidyModule:
    """
    After each SI round, redistribute a fraction of the punishment pool
    back to the top contributors as a subsidy.
    """

    def compute_subsidies(self, agents, round_contributions: dict) -> dict:
        """
        Returns {agent_id: subsidy_tokens}.
        
        Top SUBSIDY_TOP_N contributors in SI split a pool of
        SUBSIDY_FRACTION * total_punishment_cost tokens.
        """
        if not parameters.SUBSIDY_ENABLED or not round_contributions:
            return {}

        # Filter to only include agents in the Sanctioning Institution (SI)
        si_agent_ids = {a.agent_id for a in agents if getattr(a, 'institution_choice', 'SFI') == 'SI'}
        si_contributions = {aid: contrib for aid, contrib in round_contributions.items() if aid in si_agent_ids}
        if not si_contributions:
            return {}

        # Sort by contribution amount among SI members
        sorted_agents = sorted(si_contributions.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_agents[:parameters.SUBSIDY_TOP_N]
        
        # Calculate pool from SI punishment costs
        total_punishments_this_round = sum(
            sum(a.assigned_punishments.values())
            for a in agents if hasattr(a, 'assigned_punishments') and a.agent_id in si_agent_ids
        )
        pool = int(total_punishments_this_round * parameters.PUNISHMENT_COST * parameters.SUBSIDY_FRACTION)
        
        if pool <= 0 or not top_n:
            return {}
        
        per_agent = pool // len(top_n)
        return {aid: per_agent for aid, _ in top_n}
