"""Non-private and differentially private LoRA training for DistilGPT2.

Non-private smoke test:
    python -m src.train --max-train-samples 16 --max-train-steps 1

Private smoke test:
    python -m src.train --dp --target-epsilon 8 --max-train-samples 16 \
      --max-train-steps 2 --epochs 1 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data import MODEL_NAME, configure_tokenizer, load_e2e_splits


def set_seed(seed: int) -> None:
    """Seed every random source used by this experiment."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DistilGPT2 LoRA, optionally with DP-SGD.")
    parser.add_argument("--dp", action="store_true", help="Enable DP-LoRA through Opacus.")
    parser.add_argument("--target-epsilon", type=float, default=None)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--accountant", choices=["prv", "rdp", "gdp"], default="prv")
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lora_smoke"))
    parser.add_argument("--result-file", type=Path, default=Path("results/lora_smoke.json"))
    return parser.parse_args()


def _select_samples(dataset: Any, maximum: int):
    if maximum < 0:
        raise ValueError("max_train_samples must be zero or positive")
    return dataset if maximum == 0 else dataset.select(range(min(maximum, len(dataset))))


def assert_only_lora_trainable(model: Any) -> list[str]:
    """Prevent accidental full-model DP training in the DP-LoRA experiments."""
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name in names if "lora_" not in name]
    if unexpected:
        raise RuntimeError(
            "DP-LoRA must train adapter parameters only; unexpected trainable parameters: "
            + ", ".join(unexpected[:5])
        )
    return names


def _build_lora_model(args: argparse.Namespace, tokenizer: Any):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],
        fan_in_fan_out=True,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0 or args.epochs <= 0:
        raise ValueError("batch size, accumulation steps, and epochs must be positive")
    if args.dp and args.target_epsilon is None:
        raise ValueError("--dp requires --target-epsilon")
    if args.dp and args.gradient_accumulation_steps != 1:
        raise ValueError("DP-LoRA currently requires gradient accumulation to be 1")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError("Install torch, transformers, peft, and accelerate before training.") from error

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    splits, tokenizer = load_e2e_splits(max_seq_len=args.max_seq_len)
    train_dataset = _select_samples(splits["train"], args.max_train_samples)
    configure_tokenizer(tokenizer)
    model = _build_lora_model(args, tokenizer).to(device)
    trainable_names = assert_only_lora_trainable(model)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    privacy_engine = None
    privacy_hooks = None
    if args.dp:
        from src.dp import make_private_with_budget, register_gpt2_grad_sampler

        register_gpt2_grad_sampler()
        privacy_engine, privacy_hooks, optimizer, train_loader = make_private_with_budget(
            model=model,
            optimizer=optimizer,
            dataset=train_dataset,
            batch_size=args.batch_size,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
            accountant=args.accountant,
        )
    else:
        train_dataset = train_dataset.with_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    optimizer_steps = 0
    start_time = time.perf_counter()
    for _ in range(args.epochs):
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
        if args.max_train_steps and optimizer_steps >= args.max_train_steps:
            break

    elapsed_seconds = time.perf_counter() - start_time
    reported_epsilon = (
        privacy_engine.get_epsilon(args.target_delta) if privacy_engine is not None else None
    )
    noise_multiplier = getattr(optimizer, "noise_multiplier", None)
    if privacy_hooks is not None:
        privacy_hooks.cleanup()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    peak_memory_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "method": "dp-lora" if args.dp else "lora",
        "private": args.dp,
        "target_epsilon": args.target_epsilon if args.dp else None,
        "reported_epsilon": reported_epsilon,
        "delta": args.target_delta if args.dp else None,
        "noise_multiplier": noise_multiplier,
        "clipping_norm": args.max_grad_norm if args.dp else None,
        "accountant": args.accountant if args.dp else None,
        "poisson_sampling": args.dp,
        "secure_rng": False if args.dp else None,
        "seq_len": args.max_seq_len,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_names": trainable_names,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "physical_batch_size": args.batch_size,
        "dataset_size": len(train_dataset),
        "private_steps": optimizer_steps if args.dp else None,
        "optimizer_steps": optimizer_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "bleu": None,
        "rouge_l": None,
        "perplexity": None,
        "train_loss": sum(losses) / len(losses),
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
