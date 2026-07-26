Model: distilbert/distilgpt2
Dataset: GEM/e2e_nlg
Train/validation/test: official dataset splits
Prompt format:
  <MR> {meaning_representation} <SEP> {target} <EOS>

Example:
  <MR> name[The Eagle], food[French], area[riverside] <SEP>
  The Eagle is a French restaurant by the riverside. <EOS>

Tokenizer: DistilGPT2 tokenizer
Padding token: EOS token
Default max sequence length: 128
Context-length study: 64, 128, 256
Optimizer: AdamW
Seed: 42
Primary quality metrics: BLEU and ROUGE-L
Secondary quality metric: perplexity
Generation: fixed beam-search or greedy settings for all experiments
delta: 1e-5
physical batch size: 1–4 initially
effective batch size: choose after testing gradient accumulation
epochs: 3 initially for smoke tests
clipping norms to test: 0.1, 0.5, 1.0
epsilon sweep: 0.5, 1, 2, 4, 8, 16

No one changes a shared setting for a single method without:
1. recording it in that run’s config/result file, and
2. rerunning the comparison fairly if it affects the conclusion.