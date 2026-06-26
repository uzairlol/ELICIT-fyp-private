from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")


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


def build_round_frame(data: list[dict[str, Any]], source_file: str, meta: dict[str, Any]) -> pd.DataFrame:
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
                "si_avg_contribution": safe_float(round_data.get("si_avg_contribution")),
                "sfi_avg_contribution": safe_float(round_data.get("sfi_avg_contribution")),
                "si_total_contribution": safe_float(round_data.get("si_total_contribution")),
                "sfi_total_contribution": safe_float(round_data.get("sfi_total_contribution")),
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


def build_agent_frame(data: list[dict[str, Any]], source_file: str, meta: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_data in data:
        round_number = safe_int(round_data.get("round_number"))
        agents = round_data.get("agents", {}) or {}
        for agent_id, agent_data in agents.items():
            rows.append(
                {
                    "source_file": source_file,
                    **meta,
                    "round_number": round_number,
                    "agent_id": safe_int(agent_id),
                    "institution_choice": agent_data.get("institution_choice", ""),
                    "agent_group": agent_data.get("agent_group", "unknown"),
                    "contribution": safe_float(agent_data.get("contribution")),
                    "payoff": safe_float(agent_data.get("payoff")),
                    "cumulative_payoff": safe_float(agent_data.get("cumulative_payoff")),
                    "wealth": safe_float(agent_data.get("wealth")),
                    "reputation": safe_float(agent_data.get("reputation")),
                    "received_punishments": safe_float(agent_data.get("received_punishments")),
                    "received_rewards": safe_float(agent_data.get("received_rewards")),
                    "assigned_punishments_total": sum(safe_float(v) for v in (agent_data.get("assigned_punishments", {}) or {}).values()),
                    "assigned_rewards_total": sum(safe_float(v) for v in (agent_data.get("assigned_rewards", {}) or {}).values()),
                    "climate_damage_taken_round": safe_float(agent_data.get("climate_damage_taken_round")),
                    "climate_damage_taken_cumulative": safe_float(agent_data.get("climate_damage_taken_cumulative")),
                    "ldf_contribution_round": safe_float(agent_data.get("ldf_contribution_round")),
                    "ldf_payout_round": safe_float(agent_data.get("ldf_payout_round")),
                    "net_climate_transfer_round": safe_float(agent_data.get("net_climate_transfer_round")),
                    "parsing_failures": safe_int(agent_data.get("parsing_failures")),
                    "rule_of_law_blocks": safe_int(agent_data.get("rule_of_law_blocks")),
                    "strategy": agent_data.get("strategy", ""),
                }
            )
    return pd.DataFrame(rows)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_population_dynamics(round_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(round_df["round_number"], round_df["si_population"], marker="o", label="SI population")
    ax.plot(round_df["round_number"], round_df["sfi_population"], marker="o", label="SFI population")
    ax.set_title("Institution Population Dynamics")
    ax.set_xlabel("Round")
    ax.set_ylabel("Agents")
    ax.legend()
    save_plot(output_path)


def plot_avg_contributions(round_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(round_df["round_number"], round_df["si_avg_contribution"], marker="o", label="SI avg contribution")
    ax.plot(round_df["round_number"], round_df["sfi_avg_contribution"], marker="o", label="SFI avg contribution")
    ax.set_title("Average Contributions by Institution")
    ax.set_xlabel("Round")
    ax.set_ylabel("Contribution")
    ax.legend()
    save_plot(output_path)


def plot_total_contributions(round_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(round_df["round_number"], round_df["si_total_contribution"], marker="o", label="SI total contribution")
    ax.plot(round_df["round_number"], round_df["sfi_total_contribution"], marker="o", label="SFI total contribution")
    ax.set_title("Total Contributions by Institution")
    ax.set_xlabel("Round")
    ax.set_ylabel("Tokens")
    ax.legend()
    save_plot(output_path)


def plot_cumulative_payoffs(agent_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for agent_id, group in agent_df.groupby("agent_id"):
        ax.plot(group["round_number"], group["cumulative_payoff"], linewidth=1.8, label=f"Agent {agent_id}")
    ax.set_title("Cumulative Payoff Trajectories")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative payoff")
    ax.legend(ncol=2, fontsize=8)
    save_plot(output_path)


def plot_agent_contributions(agent_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for agent_id, group in agent_df.groupby("agent_id"):
        ax.plot(group["round_number"], group["contribution"], linewidth=1.8, label=f"Agent {agent_id}")
    ax.set_title("Agent Contribution Trajectories")
    ax.set_xlabel("Round")
    ax.set_ylabel("Contribution")
    ax.legend(ncol=2, fontsize=8)
    save_plot(output_path)


def plot_reputations(agent_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for agent_id, group in agent_df.groupby("agent_id"):
        ax.plot(group["round_number"], group["reputation"], linewidth=1.8, label=f"Agent {agent_id}")
    ax.set_title("Reputation Trajectories")
    ax.set_xlabel("Round")
    ax.set_ylabel("Reputation")
    ax.legend(ncol=2, fontsize=8)
    save_plot(output_path)


def plot_punishments_received(agent_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for agent_id, group in agent_df.groupby("agent_id"):
        ax.plot(group["round_number"], group["received_punishments"], linewidth=1.8, label=f"Agent {agent_id}")
    ax.set_title("Punishments Received")
    ax.set_xlabel("Round")
    ax.set_ylabel("Punishment tokens")
    ax.legend(ncol=2, fontsize=8)
    save_plot(output_path)


def plot_punishments_assigned(agent_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for agent_id, group in agent_df.groupby("agent_id"):
        ax.plot(group["round_number"], group["assigned_punishments_total"], linewidth=1.8, label=f"Agent {agent_id}")
    ax.set_title("Punishments Assigned")
    ax.set_xlabel("Round")
    ax.set_ylabel("Punishment tokens spent")
    ax.legend(ncol=2, fontsize=8)
    save_plot(output_path)


def plot_ldf_flows(round_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(round_df["round_number"], round_df["ldf_contributions_total"], marker="o", label="LDF contributions")
    ax.plot(round_df["round_number"], round_df["ldf_payouts_total"], marker="o", label="LDF payouts")
    ax.plot(round_df["round_number"], round_df["ldf_pool_end"], marker="o", label="LDF pool end")
    ax.set_title("LDF Pool and Flow Dynamics")
    ax.set_xlabel("Round")
    ax.set_ylabel("Value")
    ax.legend()
    save_plot(output_path)


def plot_damage_coverage(round_df: pd.DataFrame, output_path: Path) -> None:
    coverage = [
        (p / g) if g > 0 else 0.0
        for g, p in zip(round_df["gross_damage_total"], round_df["ldf_payouts_total"])
    ]
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(round_df["round_number"], round_df["gross_damage_total"], marker="o", label="Gross damage")
    ax1.plot(round_df["round_number"], round_df["net_damage_total"], marker="o", label="Net damage")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Damage")
    ax2 = ax1.twinx()
    ax2.plot(round_df["round_number"], coverage, marker="o", color="black", linestyle="--", label="Coverage ratio")
    ax2.set_ylabel("Coverage ratio")
    ax1.set_title("Climate Damage and LDF Coverage")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save_plot(output_path)


def export_plots(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for json_path in iter_result_files(results_dir):
        payload = load_json(json_path)
        if not isinstance(payload, list):
            continue

        meta = parse_filename(json_path)
        source_dir = output_dir / json_path.stem
        round_df = build_round_frame(payload, json_path.name, meta)
        agent_df = build_agent_frame(payload, json_path.name, meta)

        if round_df.empty or agent_df.empty:
            continue

        plot_population_dynamics(round_df, source_dir / "01_institution_population.png")
        plot_avg_contributions(round_df, source_dir / "02_average_contributions.png")
        plot_total_contributions(round_df, source_dir / "03_total_contributions.png")
        plot_cumulative_payoffs(agent_df, source_dir / "04_cumulative_payoffs.png")
        plot_agent_contributions(agent_df, source_dir / "05_agent_contributions.png")
        plot_reputations(agent_df, source_dir / "06_reputations.png")
        plot_punishments_received(agent_df, source_dir / "07_punishments_received.png")
        plot_punishments_assigned(agent_df, source_dir / "08_punishments_assigned.png")
        plot_ldf_flows(round_df, source_dir / "09_ldf_flows.png")
        plot_damage_coverage(round_df, source_dir / "10_damage_coverage.png")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ablation plots from simulation JSON files.")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "analysis_outputs" / "plots")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    export_plots(results_dir, output_dir)
    print(f"Exported plots to {output_dir}")


if __name__ == "__main__":
    main()