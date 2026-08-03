"""DP-LoRA training for DistilGPT2 on E2E NLG via Opacus.

Run a small CPU-safe check first:
    python -m src.train_private --epsilon 8 --max-train-samples 64 --max-train-steps 4

Adds DP-SGD (per-sample gradient clipping + calibrated noise) on top of the
same LoRA setup verified by src/train.py, using Opacus's modern
PrivacyEngine.make_private_with_epsilon API.
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
from src.train import _select_samples

_BATCH_FIELDS = ("input_ids", "attention_mask", "labels")


class _TupleDataset:
    """Adapts a dict-yielding dataset to tuple-yielding.

    Opacus's DPDataLoader infers per-field shape/dtype for empty (Poisson-
    sampled) batches by iterating ``dataset[0]``, assuming a tuple/list item.
    Iterating a dict instead yields its string keys, producing a bogus
    placeholder dtype and crashing if the very first batch drawn is empty.
    """

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        example = self._dataset[index]
        return tuple(example[field] for field in _BATCH_FIELDS)


def _collate_batch(examples: list[tuple[Any, ...]]) -> dict[str, Any]:
    import torch

    columns = zip(*examples, strict=True)
    return {field: torch.stack(column) for field, column in zip(_BATCH_FIELDS, columns, strict=True)}


def set_seed(seed: int) -> None:
    """Seed every random source used in this training script."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DP-LoRA DistilGPT2 training.")
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
        "--epsilon", type=float, required=True, help="Target privacy budget (DP epsilon)."
    )
    parser.add_argument(
        "--delta", type=float, default=1e-5, help="Target delta; must be in (0, 1)."
    )
    parser.add_argument(
        "--clipping-norm",
        type=float,
        default=1.0,
        help="Per-sample gradient clipping norm (C).",
    )
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
        help="Stop after this many logical (private) optimizer steps; use 0 for no step limit.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dp_lora_smoke"))
    parser.add_argument("--result-file", type=Path, default=Path("results/dp_lora_smoke.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0 or args.epochs <= 0:
        raise ValueError("batch size, accumulation steps, and epochs must be positive")
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 < args.delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if args.clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")

    try:
        import torch
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError(
            "Install training dependencies with: python -m pip install peft accelerate opacus"
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
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    # Opacus treats this DataLoader's batch_size as the logical (Poisson-sampled)
    # batch used for noise calibration; physical micro-batching happens below via
    # BatchMemoryManager, not gradient_accumulation_steps.
    train_loader = DataLoader(
        _TupleDataset(train_dataset),
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    model = model.to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    privacy_engine = PrivacyEngine(accountant="prv")
    model, optimizer, private_train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        max_grad_norm=args.clipping_norm,
        # The noise generator's device must match the model/gradients' device
        # (torch.normal requires it), so it can't just default to CPU once
        # training moves to GPU.
        noise_generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    # Expected number of logical (Poisson-sampled) steps per epoch; exact for
    # full-epoch runs, an upper bound when --max-train-steps stops early.
    logical_steps_per_epoch = len(private_train_loader)

    model.train()
    losses: list[float] = []
    completed_epochs = 0
    physical_steps = 0
    # --max-train-steps counts logical (private) steps, but Opacus's
    # Poisson-sampled logical-batch boundaries aren't cheaply observable from
    # this loop, so approximate the cutoff in physical micro-batches instead.
    max_physical_steps = (
        args.max_train_steps * args.gradient_accumulation_steps if args.max_train_steps else 0
    )
    stopped_early = False
    start_time = time.perf_counter()
    for _ in range(args.epochs):
        print(f"Epoch {completed_epochs + 1}/{args.epochs} (logical steps: {logical_steps_per_epoch})")
        epoch_losses: list[float] = []
        with BatchMemoryManager(
            data_loader=private_train_loader,
            max_physical_batch_size=args.batch_size,
            optimizer=optimizer,
        ) as memory_safe_loader:
            for batch in memory_safe_loader:
                # Poisson sampling can draw an empty batch. If that happens
                # before any real batch has been collated, Opacus falls back
                # to a raw list of zero-length tensors instead of a dict.
                if not isinstance(batch, dict):
                    batch = dict(zip(_BATCH_FIELDS, batch, strict=True))
                if batch["input_ids"].shape[0] == 0:
                    # An empty batch contributes nothing to the clipped
                    # gradient sum, so skip it rather than forward through
                    # GPT-2: its attention_mask.view(batch_size, -1) call
                    # can't reshape a 0-element tensor (the -1 dim is
                    # ambiguous when there are no elements at all).
                    continue
                batch = {name: value.to(device) for name, value in batch.items()}
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                losses.append(loss.detach().item())
                epoch_losses.append(loss.detach().item())
                loss.backward()
                optimizer.step()
                physical_steps += 1
                if max_physical_steps and physical_steps >= max_physical_steps:
                    stopped_early = True
                    break
        completed_epochs += 1
        epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        print(f"Epoch {completed_epochs}/{args.epochs} mean loss: {epoch_loss}")
        if stopped_early:
            break

    # An estimate, not an exact count: logical-batch sizes vary stochastically
    # around effective_batch_size under Poisson sampling, any batch that
    # happened to be empty was skipped above (no real optimizer.step()), and
    # --max-train-steps stopping mid-epoch only approximates the logical-step
    # cutoff in physical micro-batches.
    optimizer_steps = (
        min(args.max_train_steps, logical_steps_per_epoch * args.epochs)
        if stopped_early
        else logical_steps_per_epoch * completed_epochs
    )

    elapsed_seconds = time.perf_counter() - start_time
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # model is wrapped in Opacus's GradSampleModule; save the underlying peft model.
    model._module.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    peak_memory_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "method": "dp-lora",
        "private": True,
        "target_epsilon": args.epsilon,
        "reported_epsilon": privacy_engine.get_epsilon(args.delta),
        "delta": args.delta,
        "noise_multiplier": optimizer.noise_multiplier,
        "clipping_norm": args.clipping_norm,
        "seq_len": args.max_seq_len,
        "trainable_parameters": trainable_parameters,
        "effective_batch_size": effective_batch_size,
        "physical_batch_size": args.batch_size,
        "dataset_size": len(train_dataset),
        "private_steps": optimizer_steps,
        "optimizer_steps": optimizer_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "bleu": None,
        "rouge_l": None,
        "perplexity": None,
        "train_loss": sum(losses) / len(losses) if losses else None,
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
