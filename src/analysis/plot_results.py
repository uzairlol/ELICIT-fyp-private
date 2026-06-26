# plot_results.py

import os
import json
import re
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_current_plot(output_dir, plot_filename):
    """Saves the current matplotlib figure in the target directory.

    The simulation-specific folder already scopes files, so we save using short,
    stable filenames to avoid Windows long-path issues.
    """
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, plot_filename)
    plt.savefig(output_path, dpi=300)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def find_scapegoat(data):
    """Finds the agent with the lowest final cumulative payoff to highlight as the scapegoat."""
    final_round = data[-1]
    lowest_payoff = float('inf')
    scapegoat_id = 0
    agents_data = final_round.get('agents', {})
    
    for str_aid, res in agents_data.items():
        if res.get('cumulative_payoff', 0) < lowest_payoff:
            lowest_payoff = res.get('cumulative_payoff', 0)
            scapegoat_id = int(str_aid)
            
    return scapegoat_id

def plot_average_contribution(data, output_dir, filename_prefix):
    rounds = []
    avg_contributions = []
    
    for round_data in data:
        rounds.append(round_data.get('round_number', 0))
        avg_contributions.append(round_data.get('si_avg_contribution', 0))
        
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, avg_contributions, marker='o', linestyle='-', color='b')
    plt.title('Average SI Group Contribution over Time')
    plt.xlabel('Round')
    plt.ylabel('Average Contribution (Tokens)')
    plt.ylim(0, 21)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds)
    plt.tight_layout()
    save_current_plot(output_dir, "_1_avg_contribution.png")
    plt.close()

def plot_institutional_population(data, output_dir, filename_prefix):
    rounds = []
    si_counts = []
    sfi_counts = []
    
    for round_data in data:
        rounds.append(round_data.get('round_number', 0))
        si_counts.append(len(round_data.get('si_members', [])))
        sfi_counts.append(len(round_data.get('sfi_members', [])))
        
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, si_counts, marker='s', linestyle='-', color='g', label='Sanctioning Institution (SI)')
    plt.plot(rounds, sfi_counts, marker='^', linestyle='-', color='r', label='Sanction-Free Institution (SFI)')
    plt.title('Institutional Population Dynamics')
    plt.xlabel('Round')
    plt.ylabel('Number of Agents')
    plt.ylim(0, max(max(si_counts, default=10), max(sfi_counts, default=10)) + 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds)
    plt.tight_layout()
    save_current_plot(output_dir, "_2_institutions.png")
    plt.close()

def plot_cumulative_payoff(data, output_dir, filename_prefix, scapegoat_id):
    agent_ids = set()
    for round_data in data:
        for str_aid in round_data.get('agents', {}).keys():
            agent_ids.add(int(str_aid))
            
    agent_ids = sorted(list(agent_ids))
    rounds = [r.get('round_number', 0) for r in data]
    
    payoffs = {aid: [] for aid in agent_ids}
    
    for round_data in data:
        agents_dict = round_data.get('agents', {})
        for aid in agent_ids:
            str_aid = str(aid)
            if str_aid in agents_dict:
                val = agents_dict[str_aid].get('cumulative_payoff', 0)
            else:
                val = payoffs[aid][-1] if payoffs[aid] else 1000
            payoffs[aid].append(val)
            
    plt.figure(figsize=(12, 8))
    for aid in agent_ids:
        linewidth = 3 if aid == scapegoat_id else 1
        linestyle = '-' if aid == scapegoat_id else '--'
        alpha = 1.0 if aid == scapegoat_id else 0.6
        label = f'Agent {aid} {"(Scapegoat)" if aid == scapegoat_id else ""}'
        
        plt.plot(rounds, payoffs[aid], marker='.', linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)
        
    plt.title('Cumulative Payoff per Agent')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Tokens')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds)
    plt.tight_layout()
    save_current_plot(output_dir, "_3_cumulative_payoffs.png")
    plt.close()

def plot_punishments_received(data, output_dir, filename_prefix, scapegoat_id):
    agent_ids = set()
    for round_data in data:
        for str_aid in round_data.get('agents', {}).keys():
            agent_ids.add(int(str_aid))
            
    agent_ids = sorted(list(agent_ids))
    rounds = [r.get('round_number', 0) for r in data]
    
    punishments = {aid: [] for aid in agent_ids}
    
    for round_data in data:
        agents_dict = round_data.get('agents', {})
        for aid in agent_ids:
            str_aid = str(aid)
            if str_aid in agents_dict:
                val = agents_dict[str_aid].get('received_punishments', 0)
            else:
                val = 0
            punishments[aid].append(val)
            
    plt.figure(figsize=(12, 8))
    for aid in agent_ids:
        linewidth = 3 if aid == scapegoat_id else 1
        linestyle = '-' if aid == scapegoat_id else '--'
        alpha = 1.0 if aid == scapegoat_id else 0.5
        label = f'Agent {aid} {"(Scapegoat)" if aid == scapegoat_id else ""}'
        
        plt.plot(rounds, punishments[aid], marker='.', linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)
        
    plt.title('Negative Tokens (Punishment) Received per Round')
    plt.xlabel('Round')
    plt.ylabel('Negative Tokens Assigned TO this Agent')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds)
    plt.tight_layout()
    save_current_plot(output_dir, "_4_punishments_received.png")
    plt.close()

def plot_agent_contributions(data, output_dir, filename_prefix, scapegoat_id):
    agent_ids = set()
    for round_data in data:
        for str_aid in round_data.get('agents', {}).keys():
            agent_ids.add(int(str_aid))
            
    agent_ids = sorted(list(agent_ids))
    rounds = [r.get('round_number', 0) for r in data]
    
    contributions = {aid: [] for aid in agent_ids}
    
    for round_data in data:
        agents_dict = round_data.get('agents', {})
        for aid in agent_ids:
            str_aid = str(aid)
            if str_aid in agents_dict:
                val = agents_dict[str_aid].get('contribution', 0)
            else:
                val = contributions[aid][-1] if contributions[aid] else 0
            contributions[aid].append(val)
            
    plt.figure(figsize=(12, 8))
    for aid in agent_ids:
        linewidth = 3 if aid == scapegoat_id else 1
        linestyle = '-' if aid == scapegoat_id else '--'
        alpha = 1.0 if aid == scapegoat_id else 0.5
        label = f'Agent {aid} {"(Scapegoat)" if aid == scapegoat_id else ""}'
        
        plt.plot(rounds, contributions[aid], marker='.', linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)
        
    plt.title('Individual Agent Contributions per Round')
    plt.xlabel('Round')
    plt.ylabel('Tokens Contributed (Max 20)')
    plt.ylim(0, 21)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds)
    plt.tight_layout()
    save_current_plot(output_dir, "_5_agent_contributions.png")
    plt.close()

def plot_agent_reputations(data, output_dir, filename_prefix, scapegoat_id):
    agent_ids = set()
    for round_data in data:
        for str_aid in round_data.get('agents', {}).keys():
            agent_ids.add(int(str_aid))
            
    agent_ids = sorted(list(agent_ids))
    rounds = [r.get('round_number', 0) for r in data]
    
    reputations = {aid: [] for aid in agent_ids}
    
    for round_data in data:
        agents_dict = round_data.get('agents', {})
        for aid in agent_ids:
            str_aid = str(aid)
            if str_aid in agents_dict:
                val = agents_dict[str_aid].get('reputation', 0)
            else:
                val = reputations[aid][-1] if reputations[aid] else 0
            reputations[aid].append(val)
            
    plt.figure(figsize=(12, 8))
    for aid in agent_ids:
        linewidth = 3 if aid == scapegoat_id else 1
        linestyle = '-' if aid == scapegoat_id else '--'
        alpha = 1.0 if aid == scapegoat_id else 0.5
        label = f'Agent {aid} {"(Scapegoat)" if aid == scapegoat_id else ""}'
        
        plt.plot(rounds, reputations[aid], marker='.', linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)
        
    plt.title('Individual Agent Reputation Scores')
    plt.xlabel('Round')
    plt.ylabel('Reputation (Max 10)')
    plt.ylim(0, 10.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds)
    plt.tight_layout()
    save_current_plot(output_dir, "_6_agent_reputations.png")
    plt.close()

def plot_assigned_punishments(data, output_dir, filename_prefix, scapegoat_id):
    agent_ids = set()
    for round_data in data:
        for str_aid in round_data.get('agents', {}).keys():
            agent_ids.add(int(str_aid))
            
    agent_ids = sorted(list(agent_ids))
    rounds = [r.get('round_number', 0) for r in data]
    
    assigned_punishments = {aid: [] for aid in agent_ids}
    
    for round_data in data:
        agents_dict = round_data.get('agents', {})
        for aid in agent_ids:
            str_aid = str(aid)
            if str_aid in agents_dict:
                # Calculate total negative tokens assigned by this agent this round
                punishments_dict = agents_dict[str_aid].get('assigned_punishments', {})
                total_assigned = sum(int(amt) for amt in punishments_dict.values())
            else:
                total_assigned = 0
            assigned_punishments[aid].append(total_assigned)
            
    plt.figure(figsize=(12, 8))
    for aid in agent_ids:
        linewidth = 3 if aid == scapegoat_id else 1
        linestyle = '-' if aid == scapegoat_id else '--'
        alpha = 1.0 if aid == scapegoat_id else 0.5
        label = f'Agent {aid} {"(Scapegoat)" if aid == scapegoat_id else ""}'
        
        plt.plot(rounds, assigned_punishments[aid], marker='.', linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)
        
    plt.title('Aggressiveness: Negative Tokens ASSIGNED by each Agent')
    plt.xlabel('Round')
    plt.ylabel('Negative Tokens Spent')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds)
    plt.tight_layout()
    save_current_plot(output_dir, "_7_assigned_punishments.png")
    plt.close()

def plot_llm_safety_metrics(data, output_dir, filename_prefix):
    rounds = []
    parsing_failures = []
    rule_of_law_blocks = []
    
    for round_data in data:
        rounds.append(round_data.get('round_number', 0))
        
        round_parsing_failures = 0
        round_rule_blocks = 0
        
        for agent_data in round_data.get('agents', {}).values():
            round_parsing_failures += agent_data.get('parsing_failures', 0)
            round_rule_blocks += agent_data.get('rule_of_law_blocks', 0)
            
        parsing_failures.append(round_parsing_failures)
        rule_of_law_blocks.append(round_rule_blocks)
        
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, parsing_failures, marker='x', linestyle='-', color='r', label='Parsing Failures (Fallbacks)')
    plt.plot(rounds, rule_of_law_blocks, marker='d', linestyle='-', color='orange', label='Rule of Law Blocks (Hallucinations)')
    plt.title('LLM Safety Metrics: Fallbacks & Hallucinations')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Occurrences Across All Agents')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rounds)
    plt.tight_layout()
    save_current_plot(output_dir, "_8_llm_safety_errors.png")
    plt.close()


def is_ldf_scenario_file(filename, data=None):
    name = filename.lower()
    if ("_scnldf_" in name) or ("_scnclimate_" in name and "_ldf" in name):
        return True
    if data:
        for r in data:
            if r.get('ldf_pool_start', 0.0) > 0.0 or r.get('ldf_contributions_total', 0.0) > 0.0 or r.get('ldf_payouts_total', 0.0) > 0.0:
                return True
            for agent_data in r.get('agents', {}).values():
                if agent_data.get('wealth', 0.0) > 10000.0:
                    return True
    return False


def plot_ldf_pool_and_flows(data, output_dir, filename_prefix):
    rounds = [r.get('round_number', 0) for r in data]
    contrib = [r.get('ldf_contributions_total', 0.0) for r in data]
    payouts = [r.get('ldf_payouts_total', 0.0) for r in data]
    pool_end = [r.get('ldf_pool_end', 0.0) for r in data]
    shocks = [r.get('shock_occurred', False) for r in data]

    width = 0.38
    x = list(range(len(rounds)))

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.bar([i - width / 2 for i in x], contrib, width=width, color='#4C78A8', label='LDF Contributions')
    ax1.bar([i + width / 2 for i in x], payouts, width=width, color='#F58518', label='LDF Payouts')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Flow (million dollars GDP)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, pool_end, color='#54A24B', marker='o', linewidth=2.0, label='LDF Pool End')
    ax2.set_ylabel('Pool Balance (million dollars GDP)')

    for i, shock in enumerate(shocks):
        if shock:
            ax1.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.08)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    plt.title('LDF Fund Dynamics: Contributions, Payouts, Pool Balance')
    plt.tight_layout()
    save_current_plot(output_dir, "_9_ldf_pool_flows.png")
    plt.close()


def plot_ldf_damage_and_coverage(data, output_dir, filename_prefix):
    rounds = [r.get('round_number', 0) for r in data]
    gross = [r.get('gross_damage_total', 0.0) for r in data]
    net = [r.get('net_damage_total', 0.0) for r in data]
    payouts = [r.get('ldf_payouts_total', 0.0) for r in data]

    coverage = []
    for g, p in zip(gross, payouts):
        coverage.append((p / g) if g > 0 else 0.0)

    width = 0.38
    x = list(range(len(rounds)))

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.bar([i - width / 2 for i in x], gross, width=width, color='#E45756', label='Gross Damage')
    ax1.bar([i + width / 2 for i in x], net, width=width, color='#72B7B2', label='Net Damage')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Damage (million dollars GDP)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, coverage, color='#4C78A8', marker='o', linewidth=2.0, label='Coverage Ratio (Payout/Gross)')
    ax2.set_ylabel('Coverage Ratio')
    ax2.set_ylim(0, 1.05)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    plt.title('Climate Shock Impact: Gross vs Net Damage and LDF Coverage')
    plt.tight_layout()
    save_current_plot(output_dir, "_10_ldf_damage_coverage.png")
    plt.close()


def _group_round_totals(round_data):
    out = defaultdict(lambda: {'contrib': 0.0, 'payout': 0.0})
    for a in round_data.get('agents', {}).values():
        grp = a.get('agent_group', 'unknown')
        out[grp]['contrib'] += float(a.get('ldf_contribution_round', 0.0))
        out[grp]['payout'] += float(a.get('ldf_payout_round', 0.0))
    return out


def plot_ldf_group_burden_split(data, output_dir, filename_prefix):
    rounds = [r.get('round_number', 0) for r in data]
    dev_contrib = []
    deving_contrib = []
    dev_payout = []
    deving_payout = []

    for r in data:
        g = _group_round_totals(r)
        dev_contrib.append(g.get('developed', {}).get('contrib', 0.0))
        deving_contrib.append(g.get('developing', {}).get('contrib', 0.0))
        dev_payout.append(g.get('developed', {}).get('payout', 0.0))
        deving_payout.append(g.get('developing', {}).get('payout', 0.0))

    cum_dev_net = []
    cum_deving_net = []
    run_dev = 0.0
    run_deving = 0.0
    for dc, dpc, dp, dpp in zip(dev_contrib, deving_contrib, dev_payout, deving_payout):
        run_dev += dp - dc
        run_deving += dpp - dpc
        cum_dev_net.append(run_dev)
        cum_deving_net.append(run_deving)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(rounds, dev_contrib, marker='o', color='#E45756', label='Developed Contributions')
    ax1.plot(rounds, deving_contrib, marker='o', color='#72B7B2', label='Developing Contributions')
    ax1.plot(rounds, dev_payout, marker='x', linestyle='--', color='#B279A2', label='Developed Payouts')
    ax1.plot(rounds, deving_payout, marker='x', linestyle='--', color='#54A24B', label='Developing Payouts')
    ax1.set_ylabel('Per-round Flow (million dollars GDP)')
    ax1.set_title('LDF Burden Sharing by Group (Per Round)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(rounds, cum_dev_net, marker='o', color='#E45756', label='Developed Cumulative Net Transfer')
    ax2.plot(rounds, cum_deving_net, marker='o', color='#54A24B', label='Developing Cumulative Net Transfer')
    ax2.axhline(0, color='black', linewidth=1.0)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative Net (Payout - Contribution)')
    ax2.set_title('Cumulative Net Transfers by Group')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_current_plot(output_dir, "_11_ldf_group_burden.png")
    plt.close()


def _condition_from_name(filename):
    checks = ["Control", "Reputation", "Voting", "Full"]
    for c in checks:
        if f"_{c}_" in filename:
            return c
    return None


def _matching_ldf_condition_files(results_dir, selected_file):
    scenario_token = '_scnldf_' if '_scnldf_' in selected_file.lower() else '_scnclimate_'
    files = [f for f in os.listdir(results_dir) if f.endswith('.json') and scenario_token in f.lower()]

    m = re.search(rf'({re.escape(scenario_token)}.*)', selected_file, flags=re.IGNORECASE)
    if not m:
        return {}
    tail = m.group(1)

    wanted = {"Control": None, "Reputation": None, "Voting": None, "Full": None}
    for f in sorted(files):
        if tail in f:
            cond = _condition_from_name(f)
            if cond in wanted and wanted[cond] is None:
                wanted[cond] = f
    return {k: v for k, v in wanted.items() if v is not None}


def plot_ldf_condition_comparison(results_dir, selected_file, output_dir, filename_prefix):
    matches = _matching_ldf_condition_files(results_dir, selected_file)
    selected_cond = _condition_from_name(selected_file)
    if selected_cond and selected_cond not in matches:
        matches[selected_cond] = selected_file

    if len(matches) < 1:
        return False

    data_by_cond = {}
    for cond, fn in matches.items():
        data_by_cond[cond] = load_json(os.path.join(results_dir, fn))

    plt.figure(figsize=(12, 7))
    for cond in ["Control", "Reputation", "Voting", "Full"]:
        if cond not in data_by_cond:
            continue
        rounds = [r.get('round_number', 0) for r in data_by_cond[cond]]
        coop = [r.get('cooperation_rate', 0.0) for r in data_by_cond[cond]]
        plt.plot(rounds, coop, marker='o', linewidth=2, label=cond)
    plt.title('LDF Scenario: Cooperation Trajectories by Condition')
    plt.xlabel('Round')
    plt.ylabel('Cooperation Rate')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_current_plot(output_dir, "_12_ldf_coop_conditions.png")
    plt.close()

    conds = []
    mean_coop = []
    util = []
    coverage = []
    final_pool = []

    for cond in ["Control", "Reputation", "Voting", "Full"]:
        if cond not in data_by_cond:
            continue
        rounds = data_by_cond[cond]
        conds.append(cond)
        co = [r.get('cooperation_rate', 0.0) for r in rounds]
        ctot = sum(r.get('ldf_contributions_total', 0.0) for r in rounds)
        ptot = sum(r.get('ldf_payouts_total', 0.0) for r in rounds)
        gtot = sum(r.get('gross_damage_total', 0.0) for r in rounds)

        mean_coop.append(statistics.mean(co) if co else 0.0)
        util.append((ptot / ctot) if ctot > 0 else 0.0)
        coverage.append((ptot / gtot) if gtot > 0 else 0.0)
        final_pool.append(rounds[-1].get('ldf_pool_end', 0.0) if rounds else 0.0)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs[0, 0].bar(conds, mean_coop, color='#4C78A8')
    axs[0, 0].set_title('Mean Cooperation')
    axs[0, 0].set_ylim(0, 1.0)

    axs[0, 1].bar(conds, util, color='#F58518')
    axs[0, 1].set_title('LDF Utilization (Payout/Contribution)')
    axs[0, 1].set_ylim(0, 1.1)

    axs[1, 0].bar(conds, coverage, color='#54A24B')
    axs[1, 0].set_title('Damage Coverage (Payout/Gross Damage)')
    axs[1, 0].set_ylim(0, 1.1)

    axs[1, 1].bar(conds, final_pool, color='#B279A2')
    axs[1, 1].set_title('Final LDF Pool Balance')

    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('LDF Scenario Condition Comparison (Control/Reputation/Voting/Full)')
    plt.tight_layout()
    save_current_plot(output_dir, "_13_ldf_condition_kpis.png")
    plt.close()
    return True

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found. Have you run the simulation yet?")
        return
        
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return
        
    print("\n=== ELICIT Results Plotter ===")
    print("Available simulation results:")
    for i, f in enumerate(json_files):
        print(f"[{i}] {f}")
        
    try:
        selection_input = input("\nEnter the number of the simulation you want to plot: ")
        selection = int(selection_input)
        if selection < 0 or selection >= len(json_files):
            print("Invalid selection. Exiting.")
            return
    except ValueError:
        print("Please enter a valid number. Exiting.")
        return
        
    selected_file = json_files[selection]
    json_path = os.path.join(results_dir, selected_file)
    filename = selected_file.replace('.json', '')
    
    # Create specific subfolder for this simulation inside figures/
    output_dir = os.path.join(base_dir, 'figures', filename)
    ensure_dir(output_dir)
    
    print(f"\nLoading data from {selected_file}...")
    data = load_json(json_path)
    
    scapegoat_id = find_scapegoat(data)
    print(f"Identified Agent {scapegoat_id} as the scapegoat (lowest final payoff).")
    
    print(f"\nGenerating plots in '{output_dir}':")
    print(" 1. Average Group Contribution over Time (_1_avg_contribution.png)")
    print(" 2. Institutional Population Dynamics (_2_institutions.png)")
    print(f" 3. Cumulative Payoffs Tracking (_3_cumulative_payoffs.png)")
    print(f" 4. Punishments Received - Being Targeted (_4_punishments_received.png)")
    print(" 5. Individual Agent Contributions (_5_agent_contributions.png)")
    print(" 6. Individual Reputation Scores (_6_agent_reputations.png)")
    print(" 7. Aggressiveness: Punishments Assigned (_7_assigned_punishments.png)")
    print(" 8. LLM Safety: Parsing and Logic Errors (_8_llm_safety_errors.png)")
    
    plot_average_contribution(data, output_dir, filename)
    plot_institutional_population(data, output_dir, filename)
    plot_cumulative_payoff(data, output_dir, filename, scapegoat_id)
    plot_punishments_received(data, output_dir, filename, scapegoat_id)
    plot_agent_contributions(data, output_dir, filename, scapegoat_id)
    plot_agent_reputations(data, output_dir, filename, scapegoat_id)
    plot_assigned_punishments(data, output_dir, filename, scapegoat_id)
    plot_llm_safety_metrics(data, output_dir, filename)

    if is_ldf_scenario_file(selected_file, data):
        print(" 9. LDF Fund Flows and Pool Dynamics (_9_ldf_pool_flows.png)")
        print("10. Climate Damage and LDF Coverage (_10_ldf_damage_coverage.png)")
        print("11. LDF Group Burden Split (_11_ldf_group_burden.png)")
        print("12. LDF Cooperation by Condition (_12_ldf_coop_conditions.png)")
        print("13. LDF Condition KPI Dashboard (_13_ldf_condition_kpis.png)")

        plot_ldf_pool_and_flows(data, output_dir, filename)
        plot_ldf_damage_and_coverage(data, output_dir, filename)
        plot_ldf_group_burden_split(data, output_dir, filename)
        plot_ldf_condition_comparison(results_dir, selected_file, output_dir, filename)
    
    print(f"\nSuccess! Plots saved successfully.\n")

if __name__ == '__main__':
    main()
