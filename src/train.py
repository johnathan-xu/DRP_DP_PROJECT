"""Non-private LoRA smoke training for DistilGPT2 on E2E NLG.

Run a small CPU-safe check first:
    python -m src.train --max-train-samples 64 --max-train-steps 4

This script deliberately contains no DP code. It verifies the shared data
format, LoRA attachment, forward/backward pass, adapter saving, and result
logging before private training is introduced.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data import build_lora_model


def set_seed(seed: int) -> None:
    """Seed every random source used in this non-private training script."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-private DistilGPT2 LoRA training.")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=64,
        help="Use 0 for the full training split; default is a cheap smoke test.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=4,
        help="Stop after this many optimizer steps; use 0 for no step limit.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=64,
        help="Use 0 for the full validation split; default is a cheap smoke check.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lora_smoke"))
    parser.add_argument("--result-file", type=Path, default=Path("results/lora_smoke.json"))
    return parser.parse_args()


def _select_samples(dataset: Any, maximum: int):
    if maximum < 0:
        raise ValueError("max_train_samples must be zero or positive")
    if maximum == 0:
        return dataset
    return dataset.select(range(min(maximum, len(dataset))))


def _evaluate_loss(model: Any, dataset: Any, batch_size: int, device: Any) -> float | None:
    """Mean loss over `dataset` in eval mode; restores the prior train/eval mode after.

    Shared by train.py and train_private.py so validation is computed
    identically for both.
    """
    import torch
    from torch.utils.data import DataLoader

    was_training = model.training
    model.eval()
    losses: list[float] = []
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            losses.append(model(**batch).loss.item())
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else None


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0 or args.epochs <= 0:
        raise ValueError("batch size, accumulation steps, and epochs must be positive")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError(
            "Install training dependencies with: python -m pip install peft accelerate"
        ) from error

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer, splits = build_lora_model(
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    train_dataset = _select_samples(splits["train"], args.max_train_samples)
    train_dataset = train_dataset.with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = _select_samples(splits["validation"], args.max_eval_samples)
    val_dataset = val_dataset.with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    model = model.to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    optimizer_steps = 0
    validation_loss = None
    start_time = time.perf_counter()
    for epoch_index in range(args.epochs):
        for batch_index, batch in enumerate(train_loader):
            batch = {name: value.to(device) for name, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            losses.append(loss.detach().item())
            (loss / args.gradient_accumulation_steps).backward()

            is_accumulation_step = (batch_index + 1) % args.gradient_accumulation_steps == 0
            is_last_batch = batch_index + 1 == len(train_loader)
            if is_accumulation_step or is_last_batch:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if args.max_train_steps and optimizer_steps >= args.max_train_steps:
                    break
        validation_loss = _evaluate_loss(model, val_dataset, args.batch_size, device)
        print(f"Epoch {epoch_index + 1}/{args.epochs} validation loss: {validation_loss}")
        if args.max_train_steps and optimizer_steps >= args.max_train_steps:
            break

    elapsed_seconds = time.perf_counter() - start_time
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    peak_memory_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "method": "lora",
        "private": False,
        "target_epsilon": None,
        "reported_epsilon": None,
        "delta": None,
        "noise_multiplier": None,
        "clipping_norm": None,
        "seq_len": args.max_seq_len,
        "trainable_parameters": trainable_parameters,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "physical_batch_size": args.batch_size,
        "dataset_size": len(train_dataset),
        "private_steps": None,
        "optimizer_steps": optimizer_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "bleu": None,
        "rouge_l": None,
        "perplexity": None,
        "train_loss": sum(losses) / len(losses),
        "validation_loss": validation_loss,
        "train_seconds": round(elapsed_seconds, 2),
        "peak_gpu_memory_mb": peak_memory_mb,
        "device": str(device),
        "adapter_dir": str(args.output_dir),
    }
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    args.result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
