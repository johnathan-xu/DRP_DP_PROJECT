import matplotlib.pyplot as plt
import numpy as np

def plot_experiment_1_utility_retained(dp_scores):
    """Experiment 1: Percentage of Utility Retained vs E2E Standard Baseline"""
    
    # Official E2E NLG Challenge Baseline (TGen)
    metrics = ['BLEU', 'NIST', 'METEOR', 'ROUGE-L', 'CIDEr']
    tgen_baseline = [0.6593, 8.6094, 0.4483, 0.6850, 2.2338]
    
    # Calculate percentage of utility retained
    percentages = [(dp / tgen) * 100 for dp, tgen in zip(dp_scores, tgen_baseline)]
    
    # Plotting
    plt.figure(figsize=(8, 5))
    bars = plt.barh(metrics, percentages, color='mediumpurple', edgecolor='black')
    
    # Add a vertical reference line at 100% (The Standard Baseline)
    plt.axvline(x=100, color='red', linestyle='--', label='Standard E2E Baseline (100%)')
    
    # Formatting
    plt.xlabel('Utility Retained (%)')
    plt.title('Experiment 1: DP-λCGD (ε=4) vs. Standard Baseline')
    plt.xlim(0, 115) # Give space for labels
    plt.legend()
    
    # Add exact percentages to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 2, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/exp1_utility_retained.png')
    print("Saved Utility Retained chart!")

def plot_experiment_3_runtime(seq_lengths, runtimes):
    """Experiment 3: Sequence Length vs Runtime"""
    plt.figure(figsize=(6, 5))
    
    # Plot the line chart
    plt.plot(seq_lengths, runtimes, marker='o', color='tab:blue', linewidth=2, markersize=8)
    
    # Formatting
    plt.xlabel('Sequence Length (Tokens)')
    plt.ylabel('Training Runtime (Seconds)')
    plt.title('Experiment 3: Training Time vs. Context Length')
    plt.xticks(seq_lengths) # Force X-axis to only show 64 and 128
    
    # Set Y-axis limits so the line doesn't look completely flat or exaggerated
    plt.ylim(0, max(runtimes) + 2000)
    plt.grid(axis='y', alpha=0.3)
    
    # Add the exact time above each point
    for i, time_val in enumerate(runtimes):
        plt.annotate(f"{int(time_val)}s", 
                     (seq_lengths[i], runtimes[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('results/exp3_runtime.png')
    print("Saved Experiment 3 runtime chart!")

if __name__ == "__main__":
    # --- EXPERIMENT 1 ---
    dp_run_scores = [0.3174, 5.2808, 0.3617, 0.4884, 1.6965] 
    plot_experiment_1_utility_retained(dp_run_scores)
    
    # --- EXPERIMENT 3 ---
    # Only plotting the two runs we actually have
    seq_lengths = [64, 128]
    runtimes = [7942.26, 10584.75] 
    
    plot_experiment_3_runtime(seq_lengths, runtimes)