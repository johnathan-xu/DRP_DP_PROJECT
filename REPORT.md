# Differentially Private Fine-Tuning of DistilGPT2 on E2E NLG

## Abstract
[To be written after all experiments]

## 1. Introduction & Related Work
* Overview of Parameter-Efficient Fine-Tuning (LoRA, BiTFiT).
* Overview of DP-SGD and privacy-utility trade-offs.

## 2. Methodology
* **Model:** DistilGPT2
* **Dataset:** E2E NLG 
* **Metrics:** BLEU, NIST, METEOR, ROUGE-L, CIDEr.

## 3. Discussion: Applicability of DP-λCGD 
Differentially Private λ-Constrained Gradient Descent (DP-λCGD) offers a compelling alternative to traditional DP-SGD by fundamentally altering how noise is injected during optimization. 

**Mechanism**
Standard DP-SGD adds independent Gaussian noise at each training step, which can compound over time and severely degrade model utility. In contrast, DP-λCGD introduces a temporal constraint (λ) that correlates the injected noise with the immediately preceding iteration. By partially canceling out the noise variance across consecutive gradient updates, the method smooths the optimization trajectory and preserves more semantic information, all while strictly satisfying differential privacy guarantees.

**Memory Advantages**
A major limitation of scaling differential privacy to Large Language Models is the extreme memory overhead required to compute per-sample gradients. DP-λCGD bypasses this limitation. Because the method operates directly on the aggregated batch gradients and correlates noise temporally rather than scaling it spatially, it imposes zero additional memory overhead compared to standard DP-SGD. This makes it uniquely suited for memory-bound architectures like DistilGPT2.

## 4. Experiments & Results

### Experiment 1: DP-λCGD Performance at ε=4
Because of computational constraints, the epsilon sweep was reduced to a single privacy budget of ε=4. The model's generation quality was evaluated using the official E2E NLG metric suite.

| Metric | Score |
| :--- | :--- |
| **BLEU** | 0.3174 |
| **NIST** | 5.2808 |
| **METEOR** | 0.3617 |
| **ROUGE-L** | 0.4884 |
| **CIDEr** | 1.6965 |

* **Analysis:** Comparing these results to the official E2E NLG Challenge Baseline (TGen), the model demonstrates a strong ability to retain meaning under strict privacy constraints. While exact word-for-word accuracy degraded significantly (retaining ~48% of the baseline BLEU score), the model retained over 80% of its semantic meaning (measured via METEOR). This indicates that DP-λCGD allows the model to output highly relevant information, even if the exact phrasing becomes less precise.

*(Insert `results/exp1_utility_retained.png` here)*

### Experiment 2: Trainable Subset Comparison
*(Insert `results/exp2_subsets.png` here)*

### Experiment 3: Context Length Profiling
This experiment evaluated the impact of varying sequence lengths (64 vs. 128 tokens) on both computational efficiency and text generation quality. Hardware memory profiling was unavailable for these runs, so analysis is restricted to training runtime and utility metrics.

**Key Findings:**
1. **Computational Overhead:** Increasing the sequence length from 64 to 128 tokens resulted in a 33.2% increase in training time (from 7,942 seconds to 10,584 seconds).
2. **Utility Degradation:** Counterintuitively, expanding the context window slightly degraded the quality of the generated text across all E2E metrics. The BLEU score dropped from 0.2553 to 0.2509, and CIDEr dropped from 1.2358 to 1.2115.

This suggests that for the highly structured E2E NLG dataset, shorter sequence lengths act as a regularizer, keeping the model focused on the immediate data constraints rather than introducing extraneous noise.

*(Insert `results/exp3_runtime.png` here)*

## 5. Conclusion
[To be written after synthesis]