"""Generate E2E NLG outputs from a saved LoRA/DP-LoRA adapter.

Run against a smoke-trained adapter first:
    python -m src.generate --adapter-dir artifacts/dp_lora_smoke --max-eval-samples 8

Loads the base DistilGPT2 model + tokenizer the same way training did, attaches
a trained adapter on top (from src/train.py or src/train_private.py's
--output-dir), and writes generated continuations for a dataset split to a
JSONL file. Pass --predictions-txt to also write plain-text predictions, one
per line in dataset order, for a teammate's metric script (e.g. against the
E2E test reference file). This script only produces generations; computing
BLEU/ROUGE-L/perplexity from them is the shared eval.py/metrics.py step
EXPERIMENT_PROTOCOL.md calls for, not part of this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data import DATASET_NAME, MODEL_NAME, OFFICIAL_SPLITS, format_prompt
from src.train import _select_samples

# Fixed per EXPERIMENT_PROTOCOL.md's Generation Settings: "no per-run deviation."
# These are constants, not CLI flags, so a sweep run can't silently diverge.
MAX_NEW_TOKENS = 64
NO_REPEAT_NGRAM_SIZE = 4
NUM_BEAMS = 1  # greedy decoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate E2E outputs from a saved adapter.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=[split for split in OFFICIAL_SPLITS if split != "train"], default="validation"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=8,
        help="Use 0 for the full split; default is a cheap smoke test.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-file", type=Path, default=Path("results/generations.jsonl"))
    parser.add_argument(
        "--predictions-txt",
        type=Path,
        default=None,
        help=(
            "Optional: also write one generated prediction per line, in dataset "
            "order, for a teammate's metric script (e.g. against the E2E test "
            "reference file)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_eval_samples < 0:
        raise ValueError("max_eval_samples must be zero or positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    try:
        import torch
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Install generation dependencies with: python -m pip install peft accelerate datasets"
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    # Left-padding for batched causal-LM generation: all prompts must end at
    # the same column so generate() can append each new token uniformly
    # across the batch. Training used right-padding for teacher forcing over
    # a fixed-length sequence, which is a different concern.
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model = model.to(device)
    model.eval()

    split_data = load_dataset(DATASET_NAME, trust_remote_code=True)[args.split]
    split_data = _select_samples(split_data, args.max_eval_samples)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(split_data), args.batch_size):
            batch = split_data[start : start + args.batch_size]
            prompts = [format_prompt(mr) for mr in batch["meaning_representation"]]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for mr, target, generated in zip(
                batch["meaning_representation"], batch["target"], generated_texts, strict=True
            ):
                records.append(
                    {
                        "meaning_representation": mr,
                        "target": target,
                        "generated": generated.strip(),
                    }
                )

    with args.output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    if args.predictions_txt is not None:
        # One line per record, in the same order as split_data (no shuffling
        # anywhere in this script). This assumes the teammate's reference
        # file has one line per *raw dataset row*, not one line per unique
        # meaning representation with references grouped underneath it (the
        # official E2E NLG Challenge scoring format groups multiple human
        # references per MR). Confirm which convention their reference file
        # uses before trusting a metric score computed against this file.
        args.predictions_txt.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_txt.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record["generated"] + "\n")
        print(f"Wrote {len(records)} predictions to {args.predictions_txt}")

    print(f"Wrote {len(records)} generations to {args.output_file}")
    for record in records[:3]:
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
