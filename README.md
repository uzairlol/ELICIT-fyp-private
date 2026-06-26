# ELICIT: Emergent LLM Institutions for Climate and International Treaties

**ELICIT** (Emergent LLM Institutions for Climate and International Treaties) is a multi-agent simulation framework for studying cooperation, sanctions, governance, and climate risk-sharing under repeated public goods interactions.

Agents are LLM-driven and make decisions each round about:

- Institutional membership (Sanctioning Institution or Sanction-Free Institution)
- Contribution levels
- Punishment and reward allocations (for Sanctioning Institution members)
- Policy changes through constitutional voting

The framework also supports climate shocks and a persistent Loss & Damage Fund (LDF), with heterogeneous developed/developing profiles.

## Scope of this repository

Tracked source and sample outputs:

- `src/` — simulation and analysis code
- `results/` — simulation JSON outputs (sample runs may be committed)
- `dashboard/` — static HTML visualizer for result files

Generated locally (gitignored):

- `analysis_outputs/` — batch metrics CSVs and plot exports
- `src/debug_logs/` — per-agent LLM prompt/response debug dumps

## Core simulation model

Each round follows this high-level sequence:

1. Agents select institutions (or are routed by climate-mode rules)
2. Agents contribute in stage 1
3. Public goods returns are distributed
4. SI members assign punishments/rewards in stage 2
5. Optional subsidy redistribution is applied
6. Optional climate shock is sampled
7. Optional LDF contributions/payouts are computed
8. Round payoffs, wealth, reputation, and logs are updated
9. Optional Theory of Mind audits (batched) and gossip are applied
10. Optional democracy session runs every configured interval

## Institutions

- **SFI**: No stage-2 punish/reward actions
- **SI**: Stage-2 punish/reward is enabled

In climate/LDF mode, institution assignment is deterministic by group:

- Developed agents are routed to SI
- Developing agents are routed to SFI

## Folder structure

```text
.
├── src/
│   ├── main.py                 # Single-run entry point
│   ├── run_experiments.py      # Batch sweeps across seeds/conditions
│   ├── core/                   # Simulation engine
│   │   ├── agent.py
│   │   ├── environment.py
│   │   ├── institution.py
│   │   ├── loss_damage_fund.py
│   │   ├── subsidy.py
│   │   ├── parameters.py
│   │   ├── scenario_config.py
│   │   ├── personas.py
│   │   └── utils.py
│   ├── modules/                # Cognitive / governance modules
│   │   ├── tom_module.py
│   │   ├── gossip_module.py
│   │   ├── democracy_module.py
│   │   └── oracle.py
│   ├── prompts/                # LLM prompt construction
│   │   ├── prompt_generator.py
│   │   └── prompt_utils.py
│   ├── parsing/                # LLM response parsers
│   │   ├── institution_parser.py
│   │   ├── contribution_parser.py
│   │   ├── punishment_parser.py
│   │   └── response_parsing_utils.py
│   ├── llm/
│   │   └── ollama_client.py
│   ├── analysis/               # Post-run plotting and metrics export
│   │   ├── plot_results.py
│   │   ├── plot_wordcloud.py
│   │   ├── export_ablation_metrics.py
│   │   └── export_ablation_plots.py
│   └── debug_logs/             # Created at runtime (gitignored)
├── results/                    # Simulation JSON outputs
├── analysis_outputs/           # Generated plots/metrics (gitignored)
│   ├── metrics/
│   └── plots/
└── dashboard/                  # Browser-based result visualizer
```

## What each top-level folder contains

### `src/core`

Round orchestration, agents, institutions, payoffs, LDF/subsidy mechanics, and global configuration (`parameters.py`).

### `src/modules`

Optional cognitive and governance layers: batched ToM audits, gossip, constitutional voting, and the oracle heuristic.

### `src/prompts` and `src/parsing`

Prompt templates and JSON response parsers for institution choice, contributions, and punishments/rewards.

### `src/llm`

Local Ollama client wrapper (OpenAI-compatible API + native HTTP for reasoning models).

### `src/analysis`

Post-processing utilities. Defaults read from `results/` and write to `analysis_outputs/`.

### `results`

JSON outputs from simulation runs. Each file is a list of round snapshots with agent-level state, sanctions, reputation, climate/LDF fields, and constitutional sessions.

### `analysis_outputs`

Generated artifacts (not committed):

- `metrics/` — CSV exports from `export_ablation_metrics.py`
- `plots/` — PNG dashboards from `export_ablation_plots.py` and interactive `plot_results.py`

### `dashboard`

Static HTML/JS visualizer (`dashboard/index.html`) for exploring a results JSON file in the browser.

## Requirements

```bash
pip install -r requirements.txt
```

The framework expects a local Ollama endpoint compatible with OpenAI-style requests.

```bash
ollama pull llama3.1:8b
```

Recommended Ollama server settings for a single-GPU workstation (e.g. GTX 1070 8GB):

```powershell
$env:OLLAMA_NUM_PARALLEL = "1"
ollama serve
```

Tune GPU options in `src/core/parameters.py` (`OLLAMA_NUM_GPU`, `OLLAMA_NUM_CTX`, `OLLAMA_NUM_PARALLEL`).

## Running a single simulation

From the repository root:

```bash
python src/main.py
```

Common options:

```bash
python src/main.py --model-name llama3.1:8b
python src/main.py --scenario ldf --enable-climate-shocks --enable-ldf
python src/main.py --agent-type Random
python src/main.py --num-agents 7 --num-rounds 30
```

Notes:

- In climate/ldf scenario, agent count may auto-adjust to match configured developed/developing counts
- Results are timestamped JSON files written to `results/`
- Debug prompt dumps go to `src/debug_logs/`

## Running experiment sweeps

```bash
python src/run_experiments.py
```

Useful flags:

```bash
python src/run_experiments.py --scenario ldf --enable-climate-shocks --enable-ldf
python src/run_experiments.py --quick-compare --seeds 1
python src/run_experiments.py --full-only
python src/run_experiments.py --skip-mixed
python src/run_experiments.py --quiet
```

`run_experiments.py` resolves paths relative to the repo root automatically.

## Generating figures and metrics

Interactive per-run plots:

```bash
python src/analysis/plot_results.py
```

Batch metrics export:

```bash
python src/analysis/export_ablation_metrics.py
python src/analysis/export_ablation_plots.py
```

Optional word clouds (opens a file picker):

```bash
python src/analysis/plot_wordcloud.py
```

All analysis scripts default to:

- input: `<repo>/results/`
- output: `<repo>/analysis_outputs/`

## Configuration reference

Main tuning is in `src/core/parameters.py`.

Important groups:

- Simulation: `NUM_AGENTS`, `NUM_ROUNDS`, `SEED`, `SCENARIO`
- LLM: `LLM_MODEL`, `LLM_BASE_URL`, `LLM_MAX_CONCURRENCY`, `OLLAMA_NUM_GPU`, `OLLAMA_NUM_CTX`, `OLLAMA_NUM_PARALLEL`
- Public goods: `ENDOWMENT_STAGE_1`, `PUBLIC_GOOD_MULTIPLIER`, punishment/reward costs and effects
- Cognition/governance: `TOM_ENABLED`, `GOSSIP_ENABLED`, `DEMOCRACY_ENABLED`, `DEMOCRACY_INTERVAL`
- Climate/LDF: `CLIMATE_SHOCK_*`, `LDF_*`, `LDF_AGENT_GROUP_COUNTS`

## Reproducibility

- Random seed controlled by `SEED` in `parameters.py` and CLI overrides
- Mixed-population assignment uses deterministic shuffling under the configured seed

## Suggested GitHub description

ELICIT: LLM-based multi-agent simulation of public goods cooperation with sanctions, reputation, constitutional voting, and climate loss-and-damage risk-sharing.
