Experiment Protocol — DP Fine-Tuning of DistilGPT2

This file is the single source of truth for shared settings across all experiments. No one changes a shared setting for a single method without:

recording it in that run's config/result file, and
rerunning the comparison fairly if it affects the conclusion.
Model & Data
Model: distilbert/distilgpt2
Dataset: GEM/e2e_nlg
Splits: official E2E splits: train, validation, test. Challenge/sample splits are not used.
Test data is reserved for final reported numbers only. All tuning (LoRA rank, learning rate, clip norm, epochs, generation settings) uses the validation split.
Prompt Format
<MR> {meaning_representation} <SEP> {target}{tokenizer.eos_token}

Example:

<MR> name[The Eagle], food[French], area[riverside] <SEP> The Eagle is a French restaurant by the riverside.<|endoftext|>
<MR> and <SEP> are added as special tokens to the tokenizer vocabulary (not left as plain text), via tokenizer.add_special_tokens(...). Model embeddings must be resized (model.resize_token_embeddings(len(tokenizer))) after adding tokens. Everyone must do this identically or token IDs will not match across code. The tokenizer's existing EOS token is appended to every target; do not add a second literal <EOS> token.
Tokenizer & Sequence Handling
Tokenizer: DistilGPT2 tokenizer (AutoTokenizer.from_pretrained("distilbert/distilgpt2"))
Padding token: EOS token (tokenizer.pad_token = tokenizer.eos_token)
Padding side: right-padding (default) — confirm this is set explicitly, not left to default in case a library changes it
Default max sequence length: 128
Context-length study values: 64, 128, 256 (Phase 4 only — all other experiments use 128)
Label Masking

Prompt tokens (everything through and including <SEP>) are masked out of the loss. Only the target sentence and the tokenizer's existing EOS token contribute to loss.

Input IDs:  <MR> ...meaning representation... <SEP>  The Eagle is a French restaurant...  <tokenizer EOS>
Labels:     -100  -100 ... -100              -100    The Eagle is a French restaurant...  <tokenizer EOS>

-100 is PyTorch's ignore-index for CrossEntropyLoss — the model is trained to generate the target from the MR, not to reconstruct the prompt.

Training Setup
Optimizer: AdamW
Learning rate: 5e-4 (tentative — tune on validation split, record final value used per run in that run's config)
Seed: 42 (all runs, all sources of randomness: Python random, NumPy, PyTorch, and the DP noise generator)
Epochs: 3 initially, for smoke tests (revisit once full runs are feasible; record actual epochs per run)
LoRA Configuration (non-private baseline & DP-LoRA)
Rank (r): 8
Alpha: 16
Dropout: 0.0
Target modules (full baseline / full DP-LoRA): c_attn, c_proj
DistilGPT2 has 6 transformer blocks (indices 0–5). "Last block only" in the subset study means block index 5 — LoRA applied only to c_attn/c_proj within that block.
LM-head only: LoRA (or direct fine-tuning, decide and record which) applied only to lm_head.
Differential Privacy Settings
Delta (δ): 1e-5
Privacy accountant: PRV accountant (confirm this is what your Opacus version defaults to, and pin the Opacus version below — do not mix accountants across runs being compared)
Physical batch size: 1–4 initially (constrained by GPU memory; record actual value per run)
Effective batch size: determined after testing gradient accumulation; record final value here once fixed: ___
Clipping norms to test: 0.1, 0.5, 1.0 (select best on validation split; record chosen value per run)
Epsilon sweep: 0.5, 1, 2, 4, 8, 16
Shared epsilon for subset/context studies: ε = 4 (unless a run explicitly deviates and records why)
Generation Settings (fixed for ALL experiments — no per-run deviation)
Decoding method: greedy decoding (if the group prefers beam search instead, replace this line with: beam search, num_beams=5 — pick ONE, do not mix across runs)
max_new_tokens: 64
no_repeat_ngram_size: 4
Other generation kwargs: none beyond the above unless explicitly agreed and recorded here
Metrics
Primary: BLEU, ROUGE-L
Secondary: Perplexity
Computed identically for every run using the shared eval.py / metrics.py — no per-person metric scripts.
Software Versions (pin once decided, record here)
transformers==5.14.1
peft==0.20.0
opacus==___
dp-transformers==___
fastDP (awslabs)==___ or commit hash: ___
Result Schema (every run logs exactly this)
json
{
  "method": "lora | dp-lora | dp-bitfit",
  "private": false,
  "target_epsilon": null,
  "reported_epsilon": null,
  "delta": 1e-5,
  "noise_multiplier": null,
  "clipping_norm": null,
  "seq_len": 128,
  "trainable_parameters": 0,
  "effective_batch_size": null,
  "physical_batch_size": null,
  "dataset_size": null,
  "private_steps": null,
  "epochs": 3,
  "learning_rate": 5e-4,
  "seed": 42,
  "bleu": 0,
  "rouge_l": 0,
  "perplexity": 0,
  "train_seconds": 0,
  "peak_gpu_memory_mb": 0
}
Ground Rule

No one changes a shared setting for a single method without:

recording it in that run's config/result file, and
rerunning the comparison fairly if it affects the conclusion.
