import os
import json
import math
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def get_institution_text(agent_data):
    texts = []
    val = agent_data.get('institution_reasoning', "")
    if val and isinstance(val, str):
        texts.append(val)
    facts = agent_data.get('institution_facts_used', [])
    if isinstance(facts, list):
        for f in facts:
            if isinstance(f, str):
                texts.append(f)
    return " ".join(texts).strip()

def get_contribution_text(agent_data):
    texts = []
    val = agent_data.get('contribution_reasoning', "")
    if val and isinstance(val, str):
        texts.append(val)
    facts = agent_data.get('contribution_facts_used', [])
    if isinstance(facts, list):
        for f in facts:
            if isinstance(f, str):
                texts.append(f)
    return " ".join(texts).strip()

def get_punishment_text(agent_data):
    texts = []
    for key in ['punishment_reasoning', 'deanonymized_punishment_reasoning']:
        val = agent_data.get(key, "")
        if val and isinstance(val, str):
            texts.append(val)
    facts = agent_data.get('punishment_facts_used', [])
    if isinstance(facts, list):
        for f in facts:
            if isinstance(f, str):
                texts.append(f)
    return " ".join(texts).strip()

def get_belief_text(agent_data):
    texts = []
    belief = agent_data.get('belief_state')
    if isinstance(belief, dict):
        for key in ['institutional_strategy', 'observations']:
            val = belief.get(key, "")
            if val and isinstance(val, str):
                texts.append(val)
        trust = belief.get('trust_levels')
        if isinstance(trust, dict):
            for t_val in trust.values():
                if t_val and isinstance(t_val, str):
                    texts.append(t_val)
    return " ".join(texts).strip()

def get_aggregated_text(agent_data):
    return " ".join([
        get_institution_text(agent_data),
        get_contribution_text(agent_data),
        get_punishment_text(agent_data),
        get_belief_text(agent_data)
    ]).strip()

def generate_wordcloud_grid(data, round_num, total_rounds, agent_ids, num_agents, category_name, text_extractor, output_dir):
    # Determine grid size
    if num_agents == 7:
        cols, rows = 4, 2
    elif num_agents == 26:
        cols, rows = 6, 5
    else:
        cols = math.ceil(math.sqrt(num_agents))
        rows = math.ceil(num_agents / cols)
        
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    
    # Flatten axes for easy indexing
    if isinstance(axes, plt.Axes):
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    for i, agent_id in enumerate(agent_ids):
        ax = axes_flat[i]
        agent_data = data['agents'][agent_id]
        text = text_extractor(agent_data)
        
        # If agent has no words, generate a simple placeholder word cloud
        if not text:
            text = f"Agent_{agent_id}_No_Data"
            
        try:
            wordcloud = WordCloud(
                width=400,
                height=300,
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate(text)
            
            ax.imshow(wordcloud, interpolation='bilinear')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error generating\nword cloud:\n{str(e)}", 
                    ha='center', va='center', fontsize=8, color='red')
                    
        ax.axis('off')
        
        # Customize title with strategy or agent group information if available
        group = agent_data.get('agent_group', '')
        title_text = f"Agent {agent_id}"
        if group:
            title_text += f" ({group})"
        ax.set_title(title_text, fontsize=9, fontweight='bold')
        
    # Turn off/hide unused axes
    for i in range(len(agent_ids), rows * cols):
        fig.delaxes(axes_flat[i])
        
    display_category = category_name.replace("_", " ").title()
    plt.suptitle(f"Word Clouds ({display_category}) - Round {round_num}", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"round_{round_num:02d}_{category_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Set up Tkinter file dialog
    root = tk.Tk()
    root.withdraw()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return
        
    print("Opening file dialog to select simulation results file...")
    file_path = filedialog.askopenfilename(
        initialdir=results_dir,
        title="Select results JSON file",
        filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return
        
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    filename = os.path.basename(file_path).replace('.json', '')
    output_dir = os.path.join(base_dir, 'figures', f"{filename}_wordclouds")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each round
    total_rounds = len(data)
    print(f"Total rounds: {total_rounds}")
    
    categories = {
        'aggregated': get_aggregated_text,
        'institution_reasoning': get_institution_text,
        'contribution_reasoning': get_contribution_text,
        'punishment_reasoning': get_punishment_text,
        'belief_state': get_belief_text
    }
    
    for round_idx, round_data in enumerate(data):
        round_num = round_data.get('round_number', round_idx + 1)
        print(f"Generating word clouds for Round {round_num}/{total_rounds}...")
        
        agents = round_data.get('agents', {})
        agent_ids = sorted(list(agents.keys()), key=lambda x: int(x) if x.isdigit() else x)
        num_agents = len(agent_ids)
        
        if num_agents == 0:
            print(f"No agents found in round {round_num}. Skipping.")
            continue
            
        for category_name, extractor in categories.items():
            generate_wordcloud_grid(
                round_data, 
                round_num, 
                total_rounds, 
                agent_ids, 
                num_agents, 
                category_name, 
                extractor, 
                output_dir
            )
            
    print(f"\nSuccessfully generated combined & separate word cloud plots for all rounds in: {output_dir}")

if __name__ == '__main__':
    main()
