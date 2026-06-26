#main.py

import argparse
import logging
import os
import glob
import random

from core.agent import Agent
from core.environment import Environment
from core import parameters
from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure root logger with a timestamped format."""
    log_dir = os.path.join(os.path.dirname(__file__), 'debug_logs')
    os.makedirs(log_dir, exist_ok=True)
    debug_log_path = os.path.join(log_dir, 'debug.log')
    
    fh, ch = logging.FileHandler(debug_log_path), logging.StreamHandler()
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[fh, ch],
        force=True
    )


def clear_debug_logs():
    """
    Clears all debug log files from the previous run so each run starts fresh.
    """
    log_dir = os.path.join(os.path.dirname(__file__), 'debug_logs')
    if os.path.exists(log_dir):
        files = glob.glob(os.path.join(log_dir, '*.json'))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        if files:
            logger.info(f"Cleared {len(files)} old debug log(s) from '{log_dir}'.")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run the public goods game simulation with local Ollama (Llama 3.1 8b)."
    )
    parser.add_argument('--num-rounds', type=int, default=parameters.NUM_ROUNDS,
                        help=f"Number of rounds for the simulation (default: {parameters.NUM_ROUNDS})")
    parser.add_argument('--num-agents', type=int, default=parameters.NUM_AGENTS,
                        help=f"Number of agents in the simulation (default: {parameters.NUM_AGENTS})")
    parser.add_argument('--model-name', type=str, default=parameters.LLM_MODEL,
                        help=f"Ollama model name (default: {parameters.LLM_MODEL})")
    parser.add_argument('--scenario', type=str, default=parameters.SCENARIO,
                        help=f"Scenario framing (abstract, ldf, tax) (default: {parameters.SCENARIO})")
    parser.add_argument('--agent-type', type=str, default=parameters.AGENT_TYPE,
                        help=f"Type of agent (LLM, Random, Greedy) (default: {parameters.AGENT_TYPE})")
    parser.add_argument('--enable-climate-shocks', action='store_true',
                        help=f"Enable climate shocks (default: {parameters.CLIMATE_SHOCK_ENABLED})")
    parser.add_argument('--enable-ldf', action='store_true',
                        help=f"Enable Loss & Damage Fund (default: {parameters.LDF_ENABLED})")

    args = parser.parse_args()

    # Override parameters with command-line values
    parameters.NUM_ROUNDS = args.num_rounds
    parameters.NUM_AGENTS = args.num_agents
    parameters.LLM_MODEL = args.model_name
    parameters.SCENARIO = args.scenario
    parameters.AGENT_TYPE = args.agent_type
    parameters.CLIMATE_SHOCK_ENABLED = bool(args.enable_climate_shocks)
    parameters.LDF_ENABLED = bool(args.enable_ldf)

    scenario_key = str(parameters.SCENARIO).lower()
    if scenario_key == "climate":
        scenario_key = "ldf"

    if scenario_key == "ldf":
        ldf_counts = getattr(parameters, 'LDF_AGENT_GROUP_COUNTS', {}) or {}
        required_agents = int(ldf_counts.get('developed', 0)) + int(ldf_counts.get('developing', 0))
        if required_agents > 0 and parameters.NUM_AGENTS != required_agents:
            logger.info(
                f"[LDF] Adjusting num_agents from {parameters.NUM_AGENTS} to {required_agents} "
                f"to match configured developed/developing country counts."
            )
            parameters.NUM_AGENTS = required_agents

    # Seed Python RNG for reproducibility
    random.seed(parameters.SEED)

    # Clear previous debug logs so each run is isolated
    clear_debug_logs()

    logger.info("--- Initializing Simulation ---")
    logger.info(f"Batch: {getattr(parameters, 'BATCH_NAME', 'Default')}")
    logger.info(f"Scenario: {parameters.SCENARIO}")
    logger.info(f"Agent Type: {parameters.AGENT_TYPE}")
    logger.info(
        f"Modules: ToM={'[ON]' if parameters.TOM_ENABLED else '[OFF]'}, "
        f"Gossip={'[ON]' if parameters.GOSSIP_ENABLED else '[OFF]'}, "
        f"Voting={'[ON]' if parameters.DEMOCRACY_ENABLED else '[OFF]'}"
    )
    logger.info(
        f"Climate modules: Shocks={'[ON]' if parameters.CLIMATE_SHOCK_ENABLED else '[OFF]'}, "
        f"LDF={'[ON]' if parameters.LDF_ENABLED else '[OFF]'}"
    )
    logger.info(f"Model: {parameters.LLM_MODEL}")
    logger.info(f"Agents: {parameters.NUM_AGENTS}")
    logger.info(f"Rounds: {parameters.NUM_ROUNDS}")
    logger.info(f"Seed: {parameters.SEED}")
    if getattr(parameters, 'MIXED_AGENT_COUNTS', None):
        logger.info(f"Mixed Agent Counts: {parameters.MIXED_AGENT_COUNTS}")
    logger.info("-------------------------------")

    # Initialize the local Ollama client
    api_client = OllamaClient(model_name=parameters.LLM_MODEL, base_url=parameters.LLM_BASE_URL)

    # Initialize agents (fresh weights every run for reproducibility)
    agents = []
    mixed_counts = getattr(parameters, 'MIXED_AGENT_COUNTS', None)

    def build_group_profile_list(total_agents):
        scenario_name = str(getattr(parameters, 'SCENARIO', '')).lower()
        if scenario_name == "climate":
            scenario_name = "ldf"

        if scenario_name == "ldf":
            group_counts = getattr(parameters, 'LDF_AGENT_GROUP_COUNTS', {}) or {}
        else:
            group_counts = getattr(parameters, 'AGENT_GROUP_COUNTS', {}) or {}

        dev_n = int(group_counts.get('developed', 0))
        developing_n = int(group_counts.get('developing', 0))

        configured_total = dev_n + developing_n
        if configured_total != total_agents:
            # Proportional fallback keeps intended composition when total changes.
            if configured_total > 0:
                dev_n = int(round(total_agents * (dev_n / configured_total)))
                dev_n = max(0, min(dev_n, total_agents))
                developing_n = total_agents - dev_n
            else:
                dev_n = max(0, total_agents // 2)
                developing_n = total_agents - dev_n

        profiles = []

        if scenario_name == "ldf":
            developed_endowments = list(getattr(parameters, 'LDF_DEVELOPED_INITIAL_ENDOWMENTS', []))
            if not developed_endowments:
                developed_endowments = [float(parameters.DEVELOPED_INITIAL_WEALTH)] * dev_n
            elif len(developed_endowments) < dev_n:
                developed_endowments.extend(
                    [float(developed_endowments[-1])] * (dev_n - len(developed_endowments))
                )
            else:
                developed_endowments = developed_endowments[:dev_n]

            developing_endowments = list(getattr(parameters, 'LDF_DEVELOPING_INITIAL_ENDOWMENTS', []))
            if developing_endowments:
                if len(developing_endowments) < developing_n:
                    developing_endowments.extend(
                        [float(developing_endowments[-1])] * (developing_n - len(developing_endowments))
                    )
                else:
                    developing_endowments = developing_endowments[:developing_n]
            else:
                developing_wealth = float(
                    getattr(parameters, 'LDF_DEVELOPING_INITIAL_WEALTH', parameters.DEVELOPING_INITIAL_WEALTH)
                )
                developing_endowments = [developing_wealth] * developing_n

            for wealth in developed_endowments:
                profiles.append({'group': 'developed', 'initial_wealth': float(wealth)})
            for wealth in developing_endowments:
                profiles.append({'group': 'developing', 'initial_wealth': float(wealth)})
        else:
            for _ in range(dev_n):
                profiles.append({'group': 'developed', 'initial_wealth': float(parameters.DEVELOPED_INITIAL_WEALTH)})
            for _ in range(developing_n):
                profiles.append({'group': 'developing', 'initial_wealth': float(parameters.DEVELOPING_INITIAL_WEALTH)})

        random.shuffle(profiles)
        return profiles

    def apply_group_profile(agent, profile):
        group_name = profile.get('group', 'developing')
        initial_wealth = float(
            profile.get(
                'initial_wealth',
                parameters.DEVELOPED_INITIAL_WEALTH if group_name == 'developed' else parameters.DEVELOPING_INITIAL_WEALTH,
            )
        )

        agent.agent_group = group_name
        agent.initial_wealth = initial_wealth
        if group_name == 'developed':
            agent.wealth = initial_wealth
            agent.cumulative_payoff = initial_wealth
            agent.vulnerability = float(parameters.DEVELOPED_VULNERABILITY)
            agent.historical_emissions = float(parameters.DEVELOPED_HISTORICAL_EMISSIONS)
            agent.contribution_capacity = float(parameters.DEVELOPED_CONTRIBUTION_CAPACITY)
        else:
            agent.wealth = initial_wealth
            agent.cumulative_payoff = initial_wealth
            agent.vulnerability = float(parameters.DEVELOPING_VULNERABILITY)
            agent.historical_emissions = float(parameters.DEVELOPING_HISTORICAL_EMISSIONS)
            agent.contribution_capacity = float(parameters.DEVELOPING_CONTRIBUTION_CAPACITY)
        return agent

    def build_llm_agent(agent_id, persona="DEFAULT", profile=None):
        a = Agent(agent_id=agent_id, api_client=api_client)
        a = apply_group_profile(a, profile or {'group': 'developing'})
        a.llm_persona = persona
        if persona == "RANDOM":
            a.strategy = "LLM Random Persona"
        elif persona == "GREEDY":
            a.strategy = "LLM Greedy Persona"
        else:
            a.strategy = "LLM"
        return a

    group_profile_list = build_group_profile_list(parameters.NUM_AGENTS)

    if mixed_counts:
        llm_n = int(mixed_counts.get('LLM', 0))
        random_n = int(mixed_counts.get('Random', 0))
        greedy_n = int(mixed_counts.get('Greedy', 0))

        if llm_n + random_n + greedy_n != parameters.NUM_AGENTS:
            raise ValueError(
                f"MIXED_AGENT_COUNTS must sum to NUM_AGENTS={parameters.NUM_AGENTS}, "
                f"got {llm_n + random_n + greedy_n}"
            )

        # Deterministic assignment under seeded RNG for reproducible mixtures.
        type_list = (["LLM"] * llm_n) + (["RANDOM"] * random_n) + (["GREEDY"] * greedy_n)
        random.shuffle(type_list)

        for i, agent_kind in enumerate(type_list):
            profile = group_profile_list[i]
            if agent_kind == "RANDOM":
                agents.append(build_llm_agent(i, persona="RANDOM", profile=profile))
            elif agent_kind == "GREEDY":
                agents.append(build_llm_agent(i, persona="GREEDY", profile=profile))
            else:
                agents.append(build_llm_agent(i, persona="DEFAULT", profile=profile))
    else:
        for i in range(parameters.NUM_AGENTS):
            profile = group_profile_list[i]
            if parameters.AGENT_TYPE.upper() == "RANDOM":
                agents.append(build_llm_agent(i, persona="RANDOM", profile=profile))
            elif parameters.AGENT_TYPE.upper() == "GREEDY":
                agents.append(build_llm_agent(i, persona="GREEDY", profile=profile))
            else:
                agents.append(build_llm_agent(i, persona="DEFAULT", profile=profile))

    # Initialize and run the environment
    env = Environment(agents)
    env.run_simulation()

    # Save results to a timestamped file in the results/ folder
    if parameters.SAVE_RESULTS:
        env.save_results(
            model_name=parameters.LLM_MODEL,
            num_agents=parameters.NUM_AGENTS,
            num_rounds=parameters.NUM_ROUNDS
        )

    logger.info(f"Total LLM Cost (Local {parameters.LLM_MODEL}): ${api_client.get_total_cost():.6f}")


if __name__ == "__main__":
    main()
