import matplotlib.pyplot as plt

def plot_experiment_1(epsilons, bleu_scores, lora_baseline):
    """Experiment 1: Epsilon sweep vs Utility (BLEU)"""
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, bleu_scores, marker='o', linestyle='-', label='DP-λCGD')
    plt.axhline(y=lora_baseline, color='r', linestyle='--', label='Non-Private LoRA')
    
    plt.xscale('log', base=2)
    plt.xticks(epsilons, labels=[str(e) for e in epsilons])
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('BLEU Score')
    plt.title('Experiment 1: Privacy vs. Utility Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/exp1_utility.png')
    print("Saved Experiment 1 plot.")

# ... (We will add the plotting functions for Exp 2 and 3 later when you get that data) ...

if __name__ == "__main__":
    # Exp 1 Data
    epsilons = [0.5, 1, 2, 4, 8, 16]
    bleu_scores = [0, 0, 0, 0, 0, 0] # Waiting on Person 1 for these!
    lora_baseline = 30.7624 # Updated from evaluation_job272256.json
    
    plot_experiment_1(epsilons, bleu_scores, lora_baseline)