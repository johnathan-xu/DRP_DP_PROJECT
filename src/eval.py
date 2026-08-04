"""Generate E2E descriptions from a saved LoRA adapter and score them.

Example (safe CPU check):
    python -m src.eval --adapter-dir artifacts/lora_smoke --max-examples 10
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from src.data import DATASET_NAME, MODEL_NAME, configure_tokenizer, format_prompt, load_e2e_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate a LoRA adapter on E2E NLG.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--method", default="lora", help="Method label recorded with metrics.")
    parser.add_argument("--private", action="store_true", help="Mark this evaluation as a DP run.")
    parser.add_argument("--target-epsilon", type=float, default=None)
    parser.add_argument(
        "--training-result",
        type=Path,
        default=None,
        help="Optional training JSON to merge privacy/runtime metadata into this result row.",
    )
    parser.add_argument("--max-examples", type=int, default=100, help="Use 0 for all test examples.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-file", type=Path, default=Path("results/predictions.jsonl"))
    parser.add_argument("--result-file", type=Path, default=Path("results/evaluation.json"))
    return parser.parse_args()


def calculate_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute corpus BLEU and mean ROUGE-L F1 using the shared metric setup."""
    if not predictions or len(predictions) != len(references):
        raise ValueError("predictions and references must be non-empty lists of equal length")
    try:
        import sacrebleu
    except ImportError as error:
        raise ImportError(
            "Install evaluation dependencies with: python -m pip install sacrebleu"
        ) from error

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    rouge_l = sum(
        rouge_l_f1(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(predictions)
    return {"bleu": round(bleu, 4), "rouge_l": round(rouge_l, 6)}


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Return token-level ROUGE-L F1 using a longest-common-subsequence match.

    Keeping this compact implementation in-repo makes the metric reproducible
    across machines and avoids NLTK's optional runtime resources. The same
    function is used for every experiment, which is what comparisons require.
    """
    prediction_tokens = prediction.casefold().split()
    reference_tokens = reference.casefold().split()
    if not prediction_tokens or not reference_tokens:
        return 0.0
    previous = [0] * (len(reference_tokens) + 1)
    for prediction_token in prediction_tokens:
        current = [0]
        for index, reference_token in enumerate(reference_tokens, start=1):
            if prediction_token == reference_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs_length = previous[-1]
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _limit(dataset: Any, maximum: int):
    if maximum < 0:
        raise ValueError("max_examples must be zero or positive")
    return dataset if maximum == 0 else dataset.select(range(min(maximum, len(dataset))))


def load_adapter(adapter_dir: Path):
    """Reconstruct the base model plus saved adapter and matching tokenizer."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Install torch, transformers, and peft before evaluation.") from error

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    configure_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, adapter_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tokenizer, device


def generate_predictions(model, tokenizer, device, raw_test, max_new_tokens: int):
    """Generate one text per MR, using the protocol's fixed greedy decoding."""
    import torch

    predictions: list[str] = []
    references: list[str] = []
    records: list[dict[str, str]] = []
    for row in raw_test:
        prompt = format_prompt(row["meaning_representation"])
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        prompt_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        prediction = tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()
        reference = row["target"].strip()
        predictions.append(prediction)
        references.append(reference)
        records.append(
            {
                "meaning_representation": row["meaning_representation"],
                "reference": reference,
                "prediction": prediction,
            }
        )
    return predictions, references, records


def evaluate_perplexity(model, tokenized_test, device, batch_size: int) -> float:
    """Calculate token-weighted causal-LM perplexity on the held-out split."""
    import torch
    from torch.utils.data import DataLoader

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = tokenized_test.with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = {name: value.to(device) for name, value in batch.items()}
            loss = model(**batch).loss
            token_count = (batch["labels"][:, 1:] != -100).sum().item()
            total_nll += loss.item() * token_count
            total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("No supervised test tokens are available for perplexity.")
    return round(math.exp(total_nll / total_tokens), 6)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    start_time = time.perf_counter()
    model, tokenizer, device = load_adapter(args.adapter_dir)

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Install datasets before evaluation.") from error
    raw_test = _limit(
        load_dataset(DATASET_NAME, trust_remote_code=True)["test"], args.max_examples
    )
    predictions, references, records = generate_predictions(
        model, tokenizer, device, raw_test, args.max_new_tokens
    )
    metrics = calculate_metrics(predictions, references)

    # Perplexity uses the same fully masked, shared data pipeline as training.
    tokenized_splits, _ = load_e2e_splits(max_seq_len=args.max_seq_len)
    evaluation_test = _limit(tokenized_splits["test"], args.max_examples)
    metrics["perplexity"] = evaluate_perplexity(
        model, evaluation_test, device, args.batch_size
    )
    metrics.update(
        {
            "method": args.method,
            "private": args.private,
            "target_epsilon": args.target_epsilon,
            "adapter_dir": str(args.adapter_dir),
            "split": "test",
            "examples": len(raw_test),
            "seq_len": args.max_seq_len,
            "generation": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "eval_seconds": round(time.perf_counter() - start_time, 2),
            "device": str(device),
        }
    )
    if args.training_result is not None:
        training_metadata = json.loads(args.training_result.read_text(encoding="utf-8"))
        for key in (
            "target_epsilon",
            "reported_epsilon",
            "delta",
            "noise_multiplier",
            "clipping_norm",
            "accountant",
            "poisson_sampling",
            "secure_rng",
            "trainable_parameters",
            "effective_batch_size",
            "physical_batch_size",
            "dataset_size",
            "private_steps",
            "optimizer_steps",
            "epochs",
            "learning_rate",
            "seed",
            "train_loss",
            "train_seconds",
            "peak_gpu_memory_mb",
        ):
            if key in training_metadata:
                metrics[key] = training_metadata[key]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    args.result_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
