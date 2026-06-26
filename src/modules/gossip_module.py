# gossip_module.py

from core import parameters

class GossipModule:
    """
    Handles the aggregation and distribution of 'Social Gossip' derived from the 
    Theory of Mind (ToM) audits. Converts private trust-scores into public social pressure.
    """
    def __init__(self):
        # A bulletin of the most critical gossip from the latest round.
        # List of dicts: {'source': int, 'target': int, 'score': float, 'reasoning': str}
        self.gossip_bulletin = [] 

    def compile_gossip(self, all_audits):
        """
        Takes all ToM audits from the round and compiles the most critical gossip.
        all_audits: list of dicts: [{'source': id, 'target': id, 'score': float, 'reasoning': str}]
        """
        if not parameters.GOSSIP_ENABLED:
            return []

        # Filter for negative gossip (e.g., score <= GOSSIP_TRIGGER_SCORE)
        negative_audits = [a for a in all_audits if a['score'] <= parameters.GOSSIP_TRIGGER_SCORE]
        
        # Sort by score ascending (lowest score/least consistent first)
        negative_audits.sort(key=lambda x: x['score'])
        
        # Take the top N (most severe)
        self.gossip_bulletin = negative_audits[:parameters.MAX_GOSSIP_ITEMS]
        return self.gossip_bulletin

    def get_gossip_for_agent(self, agent):
        """
        Formats the current gossip bulletin for a specific receiving agent,
        translating actual IDs to the agent's expected pseudonyms so the agent
        can consistently track who is being talked about without breaking anonymity.
        """
        if not self.gossip_bulletin:
            return ""

        gossip_lines = []
        for gossip in self.gossip_bulletin:
            # Don't show gossip where the agent is the source (they already know what they said)
            if gossip['source'] == agent.agent_id:
                continue

            # If the receiver is the target, say "Someone said about YOU"
            if gossip['target'] == agent.agent_id:
                target_str = "YOU"
            else:
                target_str = f"Agent {gossip['target']}"

            source_str = f"Agent {gossip['source']}"
            
            # The reasoning string is scrubbed of actual IDs by the LLM instructions in tom_module.
            line = f"- {source_str} observed regarding {target_str} (Consistency Score: {gossip['score']}/10). Comment: \"{gossip['reasoning']}\""
            gossip_lines.append(line)
        
        if not gossip_lines:
            return ""
            
        return "\n".join(gossip_lines)
