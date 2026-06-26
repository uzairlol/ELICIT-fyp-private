from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROUND_AGG_COLUMNS = [
    "source_file",
    "run_label",
    "scenario",
    "condition",
    "seed",
    "num_agents",
    "num_rounds",
    "round_number",
    "si_population",
    "sfi_population",
    "si_share",
    "sfi_share",
    "si_total_contribution",
    "sfi_total_contribution",
    "si_avg_contribution",
    "sfi_avg_contribution",
    "shock_occurred",
    "shock_severity",
    "gross_damage_total",
    "net_damage_total",
    "ldf_pool_start",
    "ldf_contributions_total",
    "ldf_payouts_total",
    "ldf_pool_end",
]

AGENT_COLUMNS = [
    "source_file",
    "run_label",
    "scenario",
    "condition",
    "seed",
    "num_agents",
    "num_rounds",
    "round_number",
    "agent_id",
    "institution_choice",
    "agent_group",
    "strategy",
    "contribution",
    "stage1_payoff",
    "stage2_payoff",
    "payoff",
    "cumulative_payoff",
    "wealth",
    "reputation",
    "rank_numeric",
    "rank_text",
    "received_punishments",
    "received_rewards",
    "assigned_punishments_total",
    "assigned_rewards_total",
    "climate_damage_taken_round",
    "climate_damage_taken_cumulative",
    "ldf_contribution_round",
    "ldf_payout_round",
    "net_climate_transfer_round",
    "vulnerability",
    "historical_emissions",
    "contribution_capacity",
    "parsing_failures",
    "rule_of_law_blocks",
    "subsidy",
    "trust_level_count",
    "institutional_strategy",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_rank(rank_text: Any) -> tuple[int, str]:
    if not isinstance(rank_text, str):
        return 0, ""
    match = re.search(r"(\d+)\s*out\s*of\s*(\d+)", rank_text, flags=re.IGNORECASE)
    if not match:
        return 0, rank_text
    return int(match.group(1)), rank_text


def parse_filename(path: Path) -> dict[str, Any]:
    name = path.stem
    scenario_match = re.search(r"_scn([a-zA-Z0-9]+)_sh", name)
    condition_match = re.search(r"8b_(.+?)_scn", name)
    seed_match = re.search(r"_seed(\d+)_", name)
    agents_match = re.search(r"_(\d+)agents_", name)
    rounds_match = re.search(r"_(\d+)rounds_", name)

    return {
        "run_label": name,
        "scenario": scenario_match.group(1) if scenario_match else "unknown",
        "condition": condition_match.group(1) if condition_match else "unknown",
        "seed": safe_int(seed_match.group(1) if seed_match else None, default=-1),
        "num_agents": safe_int(agents_match.group(1) if agents_match else None, default=0),
        "num_rounds": safe_int(rounds_match.group(1) if rounds_match else None, default=0),
    }


def iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.glob("*.json") if path.is_file())


def flatten_round_rows(data: list[dict[str, Any]], source_file: str, meta: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_data in data:
        si_members = round_data.get("si_members", []) or []
        sfi_members = round_data.get("sfi_members", []) or []
        total_members = len(si_members) + len(sfi_members)
        rows.append(
            {
                "source_file": source_file,
                **meta,
                "round_number": safe_int(round_data.get("round_number")),
                "si_population": len(si_members),
                "sfi_population": len(sfi_members),
                "si_share": (len(si_members) / total_members) if total_members else 0.0,
                "sfi_share": (len(sfi_members) / total_members) if total_members else 0.0,
                "si_total_contribution": safe_float(round_data.get("si_total_contribution")),
                "sfi_total_contribution": safe_float(round_data.get("sfi_total_contribution")),
                "si_avg_contribution": safe_float(round_data.get("si_avg_contribution")),
                "sfi_avg_contribution": safe_float(round_data.get("sfi_avg_contribution")),
                "shock_occurred": bool(round_data.get("shock_occurred", False)),
                "shock_severity": safe_float(round_data.get("shock_severity")),
                "gross_damage_total": safe_float(round_data.get("gross_damage_total")),
                "net_damage_total": safe_float(round_data.get("net_damage_total")),
                "ldf_pool_start": safe_float(round_data.get("ldf_pool_start")),
                "ldf_contributions_total": safe_float(round_data.get("ldf_contributions_total")),
                "ldf_payouts_total": safe_float(round_data.get("ldf_payouts_total")),
                "ldf_pool_end": safe_float(round_data.get("ldf_pool_end")),
            }
        )
    return pd.DataFrame(rows)


def flatten_agent_rows(data: list[dict[str, Any]], source_file: str, meta: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_data in data:
        round_number = safe_int(round_data.get("round_number"))
        agents = round_data.get("agents", {}) or {}
        for agent_id, agent_data in agents.items():
            trust_levels = agent_data.get("belief_state", {}).get("trust_levels", {}) or {}
            rank_numeric, rank_text = parse_rank(agent_data.get("rank"))
            assigned_punishments = agent_data.get("assigned_punishments", {}) or {}
            assigned_rewards = agent_data.get("assigned_rewards", {}) or {}
            rows.append(
                {
                    "source_file": source_file,
                    **meta,
                    "round_number": round_number,
                    "agent_id": safe_int(agent_id),
                    "institution_choice": agent_data.get("institution_choice", ""),
                    "agent_group": agent_data.get("agent_group", "unknown"),
                    "strategy": agent_data.get("strategy", ""),
                    "contribution": safe_float(agent_data.get("contribution")),
                    "stage1_payoff": safe_float(agent_data.get("stage1_payoff")),
                    "stage2_payoff": safe_float(agent_data.get("stage2_payoff")),
                    "payoff": safe_float(agent_data.get("payoff")),
                    "cumulative_payoff": safe_float(agent_data.get("cumulative_payoff")),
                    "wealth": safe_float(agent_data.get("wealth")),
                    "reputation": safe_float(agent_data.get("reputation")),
                    "rank_numeric": rank_numeric,
                    "rank_text": rank_text,
                    "received_punishments": safe_float(agent_data.get("received_punishments")),
                    "received_rewards": safe_float(agent_data.get("received_rewards")),
                    "assigned_punishments_total": sum(safe_float(value) for value in assigned_punishments.values()),
                    "assigned_rewards_total": sum(safe_float(value) for value in assigned_rewards.values()),
                    "climate_damage_taken_round": safe_float(agent_data.get("climate_damage_taken_round")),
                    "climate_damage_taken_cumulative": safe_float(agent_data.get("climate_damage_taken_cumulative")),
                    "ldf_contribution_round": safe_float(agent_data.get("ldf_contribution_round")),
                    "ldf_payout_round": safe_float(agent_data.get("ldf_payout_round")),
                    "net_climate_transfer_round": safe_float(agent_data.get("net_climate_transfer_round")),
                    "vulnerability": safe_float(agent_data.get("vulnerability")),
                    "historical_emissions": safe_float(agent_data.get("historical_emissions")),
                    "contribution_capacity": safe_float(agent_data.get("contribution_capacity")),
                    "parsing_failures": safe_int(agent_data.get("parsing_failures")),
                    "rule_of_law_blocks": safe_int(agent_data.get("rule_of_law_blocks")),
                    "subsidy": safe_float(agent_data.get("subsidy")),
                    "trust_level_count": len(trust_levels),
                    "institutional_strategy": agent_data.get("belief_state", {}).get("institutional_strategy", ""),
                }
            )
    return pd.DataFrame(rows)


def build_run_summary(round_df: pd.DataFrame, agent_df: pd.DataFrame) -> dict[str, Any]:
    if round_df.empty:
        return {}

    final_round = round_df.sort_values("round_number").iloc[-1].to_dict()
    summary = dict(final_round)

    if not agent_df.empty:
        final_agents = agent_df.sort_values(["round_number", "agent_id"]).groupby("agent_id", as_index=False).tail(1)
        switches = (
            agent_df.sort_values(["agent_id", "round_number"])
            .groupby("agent_id")["institution_choice"]
            .apply(lambda series: max(0, int((series != series.shift()).sum()) - 1))
            .sum()
        )
        summary.update(
            {
                "final_avg_payoff": safe_float(final_agents["cumulative_payoff"].mean()),
                "mean_reputation": safe_float(final_agents["reputation"].mean()),
                "mean_parsing_failures": safe_float(final_agents["parsing_failures"].mean()),
                "mean_rule_of_law_blocks": safe_float(final_agents["rule_of_law_blocks"].mean()),
                "agent_payoff_std": safe_float(final_agents["cumulative_payoff"].std(ddof=0) if len(final_agents) > 1 else 0.0),
                "agent_contribution_mean": safe_float(agent_df["contribution"].mean()),
                "agent_contribution_std": safe_float(agent_df["contribution"].std(ddof=0) if len(agent_df) > 1 else 0.0),
                "institution_switches_total": safe_int(switches),
            }
        )
    return summary


def export_metrics(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    round_dir = output_dir / "round_metrics"
    agent_dir = output_dir / "agent_metrics"
    summary_dir = output_dir / "run_summary"
    round_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []
    all_round_frames: list[pd.DataFrame] = []
    all_agent_frames: list[pd.DataFrame] = []

    for json_path in iter_result_files(results_dir):
        payload = load_json(json_path)
        if not isinstance(payload, list):
            continue

        meta = parse_filename(json_path)
        source_file = json_path.name
        round_df = flatten_round_rows(payload, source_file, meta)
        agent_df = flatten_agent_rows(payload, source_file, meta)

        if not round_df.empty:
            round_df = round_df.reindex(columns=ROUND_AGG_COLUMNS)
            round_df.to_csv(round_dir / f"{json_path.stem}_round_metrics.csv", index=False)
            all_round_frames.append(round_df)

        if not agent_df.empty:
            agent_df = agent_df.reindex(columns=AGENT_COLUMNS)
            agent_df.to_csv(agent_dir / f"{json_path.stem}_agent_metrics.csv", index=False)
            all_agent_frames.append(agent_df)

        summary = build_run_summary(round_df, agent_df)
        if summary:
            run_summaries.append(summary)

    if all_round_frames:
        pd.concat(all_round_frames, ignore_index=True).to_csv(output_dir / "all_round_metrics.csv", index=False)
    if all_agent_frames:
        pd.concat(all_agent_frames, ignore_index=True).to_csv(output_dir / "all_agent_metrics.csv", index=False)
    if run_summaries:
        pd.DataFrame(run_summaries).to_csv(summary_dir / "run_summaries.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ablation metrics from simulation JSON files.")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "analysis_outputs" / "metrics")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    export_metrics(results_dir, output_dir)
    print(f"Exported metrics to {output_dir}")


if __name__ == "__main__":
    main()