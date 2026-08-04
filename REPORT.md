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
*(Person 4 / Lahari writes this)*
* **Mechanism:** How Differentially Private λ-Constrained Gradient Descent correlates noise only with the immediately preceding iteration.
* **Memory Advantages:** Why this method requires no additional memory overhead compared to standard DP-SGD.

## 4. Experiments & Results

### Experiment 1: Privacy vs. Utility Trade-off 
*(Insert `results/exp1_utility.png` here)*

### Experiment 2: Trainable Subset Comparison
*(Insert `results/exp2_subsets.png` here)*

### Experiment 3: Context Length Profiling
*(Insert `results/exp3_profiling.png` here)*

## 5. Conclusion
[To be written after synthesis]