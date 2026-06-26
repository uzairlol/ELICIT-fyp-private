#run_experiments.py

import logging
from pathlib import Path
import argparse
import os
import sys

repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"

os.chdir(repo_root)
sys.path.insert(0, str(src_dir))

logger = logging.getLogger(__name__)

import main
from core import parameters

ABLATION_SEEDS = [1]                 # Use 1 seed for baselines - it saves compute time
MAIN_SEEDS = [1, 2, 3, 4, 5]         # Use 5 seeds for the main LDF runs for statistical rigor
ABLATION_NUM_ROUNDS = 20
MAIN_NUM_ROUNDS = 30
DEFAULT_NUM_AGENTS = 7
DEFAULT_SCENARIO = "abstract"
DEFAULT_MODEL = "llama3.1:8b"

SEEDS = MAIN_SEEDS[:] # By default, track the main seeds for legacy loops

NUM_ROUNDS = MAIN_NUM_ROUNDS
NUM_AGENTS = DEFAULT_NUM_AGENTS
SCENARIO = DEFAULT_SCENARIO
MODEL = DEFAULT_MODEL
ENABLE_CLIMATE_SHOCKS = False
ENABLE_LDF = False
VERBOSE = True

BASELINE_AGENT_TYPES = ["Random", "Greedy"]

# New mixed-population conditions (7 total agents always)
MIXED_CONDITIONS = [
    ("Full_Mixed_LLM5_Random0_Greedy2", {"LLM": 5, "Random": 0, "Greedy": 2}),
    ("Full_Mixed_LLM5_Random2_Greedy0", {"LLM": 5, "Random": 2, "Greedy": 0}),
    ("Full_Mixed_LLM5_Random1_Greedy1", {"LLM": 5, "Random": 1, "Greedy": 1}),
]

TOTAL_RUNS = 0

current_run = 0


def _build_mixed_lookup():
    return {name: counts for name, counts in MIXED_CONDITIONS}


def _compose_batch_name(base_name):
    scenario_tag = f"scn{SCENARIO}"
    shocks_tag = "sh1" if ENABLE_CLIMATE_SHOCKS else "sh0"
    ldf_tag = "ldf1" if ENABLE_LDF else "ldf0"
    return f"{base_name}_{scenario_tag}_{shocks_tag}_{ldf_tag}"


def _compute_total_runs(seeds, include_ablations, include_mixed, full_only=False):
    total = 0
    # Ablations use only the first seed (if available) to save compute
    ablation_len = 1 if len(seeds) > 0 else 0
    
    if include_ablations:
        total += ablation_len * (1 if full_only else 4)
        if not full_only:
            total += ablation_len * len(BASELINE_AGENT_TYPES)
    if include_mixed:
        total += ablation_len * len(MIXED_CONDITIONS)
    return total


def _parse_args():
    parser = argparse.ArgumentParser(description="Run ELICIT experiment sweeps.")
    parser.add_argument("--seeds", type=int, nargs="+", default=MAIN_SEEDS,
                        help="List of random seeds for the main runs")
    parser.add_argument("--num-rounds", type=int, default=MAIN_NUM_ROUNDS,
                        help="Rounds per run (defaults to MAIN_NUM_ROUNDS)")
    parser.add_argument("--num-agents", type=int, default=DEFAULT_NUM_AGENTS,
                        help="Agents per run")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO,
                        help="Scenario name (abstract, ldf, tax)")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL,
                        help="LLM model name")
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip Control/Reputation/Voting/Full sweeps")
    parser.add_argument("--skip-mixed", action="store_true",
                        help="Skip mixed-population sweeps")
    parser.add_argument("--quick-compare", action="store_true",
                        help="Run one-seed quick compare: Full + random-mixed + greedy-mixed")
    parser.add_argument("--enable-climate-shocks", action="store_true",
                        help="Enable climate shocks for all runs")
    parser.add_argument("--enable-ldf", action="store_true",
                        help="Enable Loss & Damage Fund for all runs")
    parser.add_argument("--sweep-all", action="store_true",
                        help="In LDF mode, run the full ablation/baseline/mixed sweep instead of the main Full seed sweep")
    parser.add_argument("--full-only", action="store_true",
                        help="Run only the Full ablation (ToM + Gossip + Voting)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce run output (disables verbose simulation logs)")
    return parser.parse_args()


def run_simulation(
    batch_name,
    seed,
    tom_enabled,
    gossip_enabled,
    democracy_enabled,
    agent_type="LLM",
    mixed_agent_counts=None,
    num_rounds=None,
):
    import importlib
    import core.parameters
    importlib.reload(core.parameters)

    global current_run
    current_run += 1
    parameters.CURRENT_RUN = current_run
    parameters.TOTAL_RUNS = TOTAL_RUNS

    parameters.SEED = seed
    parameters.NUM_AGENTS = NUM_AGENTS
    actual_rounds = num_rounds if num_rounds is not None else NUM_ROUNDS
    parameters.NUM_ROUNDS = actual_rounds
    parameters.LLM_MODEL = MODEL
    parameters.SCENARIO = SCENARIO
    parameters.AGENT_TYPE = agent_type
    parameters.CLIMATE_SHOCK_ENABLED = ENABLE_CLIMATE_SHOCKS
    parameters.LDF_ENABLED = ENABLE_LDF

    # Optional mixed composition, consumed by main.py
    parameters.MIXED_AGENT_COUNTS = mixed_agent_counts

    parameters.TOM_ENABLED = tom_enabled
    parameters.GOSSIP_ENABLED = gossip_enabled
    parameters.DEMOCRACY_ENABLED = democracy_enabled
    tagged_batch_name = _compose_batch_name(batch_name)
    parameters.BATCH_NAME = tagged_batch_name

    parameters.SAVE_RESULTS = True
    parameters.VERBOSE = VERBOSE

    original_argv = sys.argv[:]
    sys.argv = [
        "main.py",
        "--num-rounds", str(actual_rounds),
        "--num-agents", str(NUM_AGENTS),
        "--model-name", MODEL,
        "--scenario", SCENARIO,
        "--agent-type", agent_type,
    ]

    if ENABLE_CLIMATE_SHOCKS:
        sys.argv.append("--enable-climate-shocks")
    if ENABLE_LDF:
        sys.argv.append("--enable-ldf")

    try:
        mix_str = f" mixed={mixed_agent_counts}" if mixed_agent_counts else ""
        logger.info(f"=== RUN Batch={tagged_batch_name} seed={seed} agent_type={agent_type}{mix_str} ===")
        main.main()
    finally:
        sys.argv = original_argv


def run_standard_sweeps(include_ablations=True, include_mixed=True, full_only=False):
    if include_ablations:
        logger.info("Starting Ablation Sweeps...")
        for seed in ABLATION_SEEDS:
            if full_only:
                # Only run Full ablation
                run_simulation("Full", seed, tom_enabled=True, gossip_enabled=True, democracy_enabled=True, num_rounds=ABLATION_NUM_ROUNDS)
            else:
                # Run all 4 ablations
                run_simulation("Control", seed, tom_enabled=False, gossip_enabled=False, democracy_enabled=False, num_rounds=ABLATION_NUM_ROUNDS)
                run_simulation("Reputation", seed, tom_enabled=True, gossip_enabled=True, democracy_enabled=False, num_rounds=ABLATION_NUM_ROUNDS)
                run_simulation("Voting", seed, tom_enabled=False, gossip_enabled=False, democracy_enabled=True, num_rounds=ABLATION_NUM_ROUNDS)
                run_simulation("Full", seed, tom_enabled=True, gossip_enabled=True, democracy_enabled=True, num_rounds=ABLATION_NUM_ROUNDS)
    else:
        logger.info("Skipping Ablation Sweeps...")

    if include_ablations and not full_only:
        logger.info("Starting Baseline Sweeps (Pure Random / Greedy)...")
        for baseline in BASELINE_AGENT_TYPES:
            for seed in ABLATION_SEEDS:
                # Baselines don't need governance mechanisms enabled
                run_simulation(
                    batch_name=f"Baseline_{baseline}",
                    seed=seed,
                    tom_enabled=False,
                    gossip_enabled=False,
                    democracy_enabled=False,
                    agent_type=baseline,
                    num_rounds=ABLATION_NUM_ROUNDS
                )
    else:
        logger.info("Skipping Baseline Sweeps (Full-only mode active or --skip-ablations used)...")

    if include_mixed:
        logger.info("Starting Mixed-Population Sweeps...")
        for batch_name, mix_counts in MIXED_CONDITIONS:
            for seed in ABLATION_SEEDS:
                # Mixed runs use the Full mechanism (ToM + Gossip + Voting).
                run_simulation(
                    batch_name=batch_name,
                    seed=seed,
                    tom_enabled=True,
                    gossip_enabled=True,
                    democracy_enabled=True,
                    agent_type="LLM",
                    mixed_agent_counts=mix_counts,
                    num_rounds=MAIN_NUM_ROUNDS
                )
    else:
        logger.info("Skipping Mixed-Population Sweeps...")


def run_quick_compare(seed):
    mixed_lookup = _build_mixed_lookup()
    quick_runs = [
        ("Full", None),
        ("Full_Mixed_LLM5_Random2_Greedy0", mixed_lookup["Full_Mixed_LLM5_Random2_Greedy0"]),
        ("Full_Mixed_LLM5_Random0_Greedy2", mixed_lookup["Full_Mixed_LLM5_Random0_Greedy2"]),
    ]

    logger.info(f"Starting Quick Compare mode with seed={seed}...")
    for batch_name, mix_counts in quick_runs:
        run_simulation(
            batch_name=batch_name,
            seed=seed,
            tom_enabled=True,
            gossip_enabled=True,
            democracy_enabled=True,
            agent_type="LLM",
            mixed_agent_counts=mix_counts,
        )


def run_main_ldf_sweep(seeds):
    logger.info("Starting Main LDF Sweep (Full mechanism across all seeds)...")
    for seed in seeds:
        run_simulation(
            batch_name="Full",
            seed=seed,
            tom_enabled=True,
            gossip_enabled=True,
            democracy_enabled=True,
            agent_type="LLM",
            num_rounds=MAIN_NUM_ROUNDS,
        )


def main_cli():
    global SEEDS, MAIN_SEEDS, ABLATION_SEEDS, NUM_ROUNDS, NUM_AGENTS, SCENARIO, MODEL, ENABLE_CLIMATE_SHOCKS, ENABLE_LDF, VERBOSE, TOTAL_RUNS, current_run

    from main import setup_logging
    setup_logging()

    args = _parse_args()

    SEEDS = args.seeds
    MAIN_SEEDS = SEEDS
    ABLATION_SEEDS = [SEEDS[0]] if SEEDS else []

    NUM_ROUNDS = args.num_rounds
    NUM_AGENTS = args.num_agents
    SCENARIO = args.scenario
    MODEL = args.model_name
    ENABLE_CLIMATE_SHOCKS = bool(args.enable_climate_shocks)
    ENABLE_LDF = bool(args.enable_ldf)
    VERBOSE = not bool(args.quiet)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Stream output line-by-line in terminals so long runs stay visible.
    try:
        stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
        stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
        if callable(stdout_reconfigure):
            stdout_reconfigure(line_buffering=True)
        if callable(stderr_reconfigure):
            stderr_reconfigure(line_buffering=True)
    except Exception:
        pass

    scenario_key = str(SCENARIO).lower()
    if scenario_key == "climate":
        scenario_key = "ldf"
    if scenario_key == "ldf":
        ldf_counts = getattr(parameters, 'LDF_AGENT_GROUP_COUNTS', {}) or {}
        required_agents = int(ldf_counts.get('developed', 0)) + int(ldf_counts.get('developing', 0))
        if required_agents > 0 and NUM_AGENTS != required_agents:
            logger.info(
                f"[LDF] Adjusting agents from {NUM_AGENTS} to {required_agents} "
                f"to match configured developed/developing country counts."
            )
            NUM_AGENTS = required_agents

    current_run = 0
    use_main_ldf_sweep = scenario_key == "ldf" and ENABLE_CLIMATE_SHOCKS and ENABLE_LDF and not args.sweep_all

    if args.quick_compare:
        quick_seed = SEEDS[0]
        TOTAL_RUNS = 3
        logger.info(
            f"Configuration: quick_compare=True, seed={quick_seed}, rounds={NUM_ROUNDS}, "
            f"agents={NUM_AGENTS}, model={MODEL}, scenario={SCENARIO}, "
            f"shocks={ENABLE_CLIMATE_SHOCKS}, ldf={ENABLE_LDF}, total_runs={TOTAL_RUNS}"
        )
        run_quick_compare(quick_seed)
    elif use_main_ldf_sweep:
        TOTAL_RUNS = len(SEEDS)

        logger.info(
            f"Configuration: quick_compare=False, seeds={SEEDS}, rounds={NUM_ROUNDS}, "
            f"agents={NUM_AGENTS}, model={MODEL}, scenario={SCENARIO}, "
            f"shocks={ENABLE_CLIMATE_SHOCKS}, ldf={ENABLE_LDF}, mode=main_only, total_runs={TOTAL_RUNS}"
        )

        if TOTAL_RUNS == 0:
            logger.info("No runs selected. Adjust flags and try again.")
            return

        run_main_ldf_sweep(SEEDS)
    else:
        include_ablations = not args.skip_ablations
        include_mixed = not args.skip_mixed
        TOTAL_RUNS = _compute_total_runs(SEEDS, include_ablations, include_mixed, args.full_only)

        logger.info(
            f"Configuration: quick_compare=False, seeds={SEEDS}, rounds={NUM_ROUNDS}, "
            f"agents={NUM_AGENTS}, model={MODEL}, scenario={SCENARIO}, "
            f"shocks={ENABLE_CLIMATE_SHOCKS}, ldf={ENABLE_LDF}, mode=sweep_all, total_runs={TOTAL_RUNS}"
        )

        if TOTAL_RUNS == 0:
            logger.info("No runs selected. Adjust flags and try again.")
            return

        run_standard_sweeps(include_ablations=include_ablations, include_mixed=include_mixed, full_only=args.full_only)

    logger.info("All runs finished.")


if __name__ == "__main__":
    main_cli()