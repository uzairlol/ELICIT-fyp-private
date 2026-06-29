# gossip_module.py

import logging
from core import parameters
from core.scenario_config import get_scenario_config

logger = logging.getLogger(__name__)

class GossipModule:
    """
    Handles the aggregation and distribution of 'Social Gossip' derived from the 
    Theory of Mind (ToM) audits. Converts private trust-scores into public social pressure.
    """
    def __init__(self, api_client=None):
        self.api_client = api_client
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
        if not negative_audits:
            self.gossip_bulletin = []
            return []

        # Group negative audits by source agent to ensure diversity
        by_source = {}
        for audit in negative_audits:
            source = audit['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(audit)

        # Sort each source's audits by score ascending (lowest first)
        for source in by_source:
            by_source[source].sort(key=lambda x: x['score'])

        # Select candidates round-robin
        selected_audits = []
        sources = sorted(list(by_source.keys()))
        indices = {src: 0 for src in sources}

        while len(selected_audits) < parameters.MAX_GOSSIP_ITEMS and sources:
            next_sources = []
            round_candidates = []
            for src in sources:
                idx = indices[src]
                if idx < len(by_source[src]):
                    round_candidates.append(by_source[src][idx])
                    indices[src] += 1
                    next_sources.append(src)
            
            if not round_candidates:
                break
            
            # Sort the candidates in this round-robin step by score ascending
            round_candidates.sort(key=lambda x: x['score'])
            
            for cand in round_candidates:
                if len(selected_audits) < parameters.MAX_GOSSIP_ITEMS:
                    selected_audits.append(cand)
                else:
                    break
            sources = next_sources

        # Generate standalone reasoning comments for the selected gossip bulletin items
        for audit in selected_audits:
            if self.api_client:
                stated_intent = audit.get('stated_intent', 'No reasoning provided.')
                contribution = audit.get('contribution', 0)
                endowment = audit.get('endowment', parameters.ENDOWMENT_STAGE_1)
                
                comment = self._generate_gossip_statement(
                    source_id=audit['source'],
                    target_id=audit['target'],
                    score=audit['score'],
                    stated_intent=stated_intent,
                    contribution=contribution,
                    endowment=endowment
                )
                audit['reasoning'] = comment
            else:
                audit['reasoning'] = "acted inconsistently with their stated intent"

        self.gossip_bulletin = selected_audits
        return self.gossip_bulletin

    def _generate_gossip_statement(self, source_id, target_id, score, stated_intent, contribution, endowment):
        """
        Generate a gossip statement about target agent's behavioral inconsistency.
        """
        sc = get_scenario_config(parameters.SCENARIO)
        currency_name = sc['currency_name']
        
        prompt = f"""You are formulating a piece of social gossip about Agent {target_id} based on an observation by Agent {source_id}.

Observation Details:
- Agent {target_id}'s stated intent: "{stated_intent}"
- Agent {target_id}'s actual contribution: {contribution} / {endowment} {currency_name}
- Behavioral Consistency Score assigned by Agent {source_id}: {score}/10 (where 10 is perfectly consistent, and 1 is highly inconsistent/hypocritical)

Task: Write a concise, natural third-person comment (at most 15 words) that a participant in the experiment might gossip about this behavior. 
Do NOT mention any specific agent IDs or numbers in the comment. Use terms like "the target" or "them".
Examples of comments:
- "promised to contribute fully but contributed almost nothing"
- "claimed they would cooperate but kept all their tokens"
- "stated they would help the group but defaulted on their word"

Return ONLY the plain-text comment. No JSON, no quote marks, no greeting, no explanation. Just the direct comment sentence."""

        try:
            response = self.api_client.send_request(
                model_name=self.api_client.deployment_name,
                prompt=prompt,
                max_tokens=64,
                temperature=0.7,
            )
            comment = response.strip().strip('"').strip("'").strip()
            return comment
        except Exception as e:
            logger.warning(f"Failed to generate gossip statement: {e}")
            return "acted inconsistently with their stated intent"

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
            reasoning = str(gossip.get('reasoning', '') or '').strip()
            if reasoning:
                line = (
                    f"- {source_str} observed regarding {target_str} "
                    f"(Consistency Score: {gossip['score']}/10). Comment: \"{reasoning}\""
                )
            else:
                line = (
                    f"- {source_str} observed regarding {target_str} "
                    f"(Consistency Score: {gossip['score']}/10)."
                )
            gossip_lines.append(line)
        
        if not gossip_lines:
            return ""
            
        return "\n".join(gossip_lines)
