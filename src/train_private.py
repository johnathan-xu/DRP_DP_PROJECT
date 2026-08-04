"""DP-LoRA training for DistilGPT2 on E2E NLG via Opacus.

Run a small CPU-safe check first:
    python -m src.train_private --epsilon 8 --max-train-samples 64 --max-train-steps 4

Adds DP-SGD (per-sample gradient clipping + calibrated noise) on top of the
same LoRA setup verified by src/train.py, using Opacus's modern
PrivacyEngine.make_private_with_epsilon API. Long runs checkpoint themselves
periodically (--checkpoint-every-steps) and on Ctrl+C/SIGTERM, and can be
continued later with --resume:
    python -m src.train_private --epsilon 8 --epochs 500 --output-dir artifacts/run1
    # ...interrupted or killed...
    python -m src.train_private --epsilon 8 --epochs 500 --output-dir artifacts/run1 --resume
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data import build_lora_model
from src.train import _evaluate_loss, _select_samples

_BATCH_FIELDS = ("input_ids", "attention_mask", "labels")

# Fields that must match between a checkpoint and a --resume invocation: any of
# these differing would silently change the model architecture, tokenization,
# or the privacy calibration target mid-run.
_RESUME_SENSITIVE_ARGS = (
    "max_seq_len",
    "epsilon",
    "delta",
    "clipping_norm",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "seed",
    "max_train_samples",
)


class _GracefulExit(Exception):
    """Raised from a SIGTERM handler so it can be caught alongside KeyboardInterrupt."""


def _handle_sigterm(signum: int, frame: Any) -> None:
    raise _GracefulExit()


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
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=64,
        help="Use 0 for the full validation split; default is a cheap smoke check.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dp_lora_smoke"))
    parser.add_argument("--result-file", type=Path, default=Path("results/dp_lora_smoke.json"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint at --output-dir/checkpoint.pt.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=2000,
        help="Save a resumable checkpoint every this many real optimizer steps; 0 to disable "
        "periodic checkpoints (Ctrl+C/SIGTERM still checkpoints).",
    )
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
    if args.checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be zero or positive")

    try:
        import torch
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from torch.utils.data import DataLoader
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError(
            "Install training dependencies with: "
            "python -m pip install peft accelerate opacus tensorboard"
        ) from error

    checkpoint_path = args.output_dir / "checkpoint.pt"
    resume_state: dict[str, Any] | None = None
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"--resume was passed but no checkpoint found at {checkpoint_path}"
            )
        # Lightweight peek at just the bookkeeping fields (not the multi-hundred-MB
        # model/optimizer tensors, which get (re-)loaded via privacy_engine.load_checkpoint
        # once the wrapped model/optimizer exist below).
        resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_args = resume_state["args"]
        for field in _RESUME_SENSITIVE_ARGS:
            if saved_args.get(field) != getattr(args, field):
                raise ValueError(
                    f"--resume config mismatch on {field!r}: checkpoint has "
                    f"{saved_args.get(field)!r}, this invocation has {getattr(args, field)!r}. "
                    "Resuming with a different value would silently corrupt the run."
                )
        print(
            f"Resuming from {checkpoint_path} "
            f"({resume_state['total_steps_completed']}/{resume_state['total_target_steps']} steps done)"
        )
        random.setstate(resume_state["rng_state"]["python"])
        np.random.set_state(resume_state["rng_state"]["numpy"])
        torch.set_rng_state(resume_state["rng_state"]["torch"])
    elif checkpoint_path.exists():
        raise FileExistsError(
            f"{checkpoint_path} already exists but --resume was not passed. "
            "Pass --resume to continue it, or choose a different --output-dir to start fresh."
        )
    else:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        if resume_state is not None and resume_state["rng_state"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(resume_state["rng_state"]["cuda"])

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
    val_dataset = _select_samples(splits["validation"], args.max_eval_samples)
    val_dataset = val_dataset.with_format(
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

    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    if resume_state is not None:
        noise_generator.set_state(resume_state["rng_state"]["noise_generator"])

    privacy_engine = PrivacyEngine(accountant="prv")
    if resume_state is not None:
        # Reuse the exact noise_multiplier already calibrated on the original
        # run. Calling make_private_with_epsilon again here would recalibrate
        # a *fresh* noise_multiplier as if starting over, silently discarding
        # however much privacy budget has already been spent.
        model, optimizer, private_train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=resume_state["noise_multiplier"],
            max_grad_norm=args.clipping_norm,
            poisson_sampling=True,
            noise_generator=noise_generator,
        )
        privacy_engine.load_checkpoint(path=checkpoint_path, module=model, optimizer=optimizer)
        total_target_steps = resume_state["total_target_steps"]
        loss_sum = resume_state["loss_sum"]
        loss_count = resume_state["loss_count"]
        elapsed_seconds_before = resume_state["elapsed_seconds_before"]
    else:
        model, optimizer, private_train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            epochs=args.epochs,
            max_grad_norm=args.clipping_norm,
            noise_generator=noise_generator,
        )
        # Poisson-sampled logical steps per pass over the dataset; the target
        # step budget the original run (and every future --resume of it) is
        # calibrated for.
        total_target_steps = len(private_train_loader) * args.epochs
        loss_sum = 0.0
        loss_count = 0
        elapsed_seconds_before = 0.0

    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
    # noise_multiplier is fixed for the whole run (calibrated once, never
    # recalibrated on resume), so it's a single value, not a curve.
    writer.add_scalar("dp/noise_multiplier", optimizer.noise_multiplier, 0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    def _save_checkpoint() -> None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        privacy_engine.save_checkpoint(
            path=checkpoint_path,
            module=model,
            optimizer=optimizer,
            checkpoint_dict={
                "noise_multiplier": optimizer.noise_multiplier,
                "total_steps_completed": len(privacy_engine.accountant),
                "total_target_steps": total_target_steps,
                "loss_sum": loss_sum,
                "loss_count": loss_count,
                "elapsed_seconds_before": elapsed_seconds_before + (time.perf_counter() - start_time),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "noise_generator": noise_generator.get_state(),
                },
                "args": {
                    **vars(args),
                    "output_dir": str(args.output_dir),
                    "result_file": str(args.result_file),
                },
            },
        )

    model.train()
    max_physical_steps = (
        args.max_train_steps * args.gradient_accumulation_steps if args.max_train_steps else 0
    )
    session_physical_steps = 0
    completed_epochs = 0
    validation_loss = None
    start_time = time.perf_counter()
    hit_step_limit = False
    try:
        while len(privacy_engine.accountant) < total_target_steps:
            completed_epochs += 1
            steps_before = len(privacy_engine.accountant)
            print(f"Pass {completed_epochs} (total steps so far: {steps_before}/{total_target_steps})")
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            reached_target_or_limit = False
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
                    loss_value = loss.detach().item()
                    loss_sum += loss_value
                    loss_count += 1
                    epoch_loss_sum += loss_value
                    epoch_loss_count += 1
                    loss.backward()
                    optimizer.step()
                    session_physical_steps += 1

                    total_steps_now = len(privacy_engine.accountant)
                    writer.add_scalar("loss/train_step", loss_value, total_steps_now)
                    if args.checkpoint_every_steps and total_steps_now % args.checkpoint_every_steps == 0:
                        _save_checkpoint()

                    if max_physical_steps and session_physical_steps >= max_physical_steps:
                        reached_target_or_limit = True
                        hit_step_limit = True
                        break
                    if total_steps_now >= total_target_steps:
                        reached_target_or_limit = True
                        break
            if reached_target_or_limit:
                break
            epoch_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count else None
            print(f"Pass {completed_epochs} mean loss: {epoch_loss}")
            total_steps_now = len(privacy_engine.accountant)
            if epoch_loss is not None:
                writer.add_scalar("loss/train_epoch_mean", epoch_loss, total_steps_now)
            # get_epsilon() re-walks the accountant's full step history, so it's
            # logged once per pass rather than every step to keep the overhead down.
            writer.add_scalar("dp/epsilon_so_far", privacy_engine.get_epsilon(args.delta), total_steps_now)
            # Evaluate the underlying peft model directly, bypassing Opacus's
            # GradSampleModule wrapper: no backward pass happens here, so its
            # per-sample-grad hooks would only add overhead.
            validation_loss = _evaluate_loss(model._module, val_dataset, args.batch_size, device)
            print(f"Pass {completed_epochs} validation loss: {validation_loss}")
            if validation_loss is not None:
                writer.add_scalar("loss/validation_epoch_mean", validation_loss, total_steps_now)
    except (KeyboardInterrupt, _GracefulExit):
        print("Interrupted; saving checkpoint before exit...")
        _save_checkpoint()
        writer.flush()
        writer.close()
        raise

    total_steps_completed = len(privacy_engine.accountant)
    if hit_step_limit and total_steps_completed < total_target_steps and args.resume:
        # Hitting --max-train-steps mid-resume isn't real completion: on a
        # fresh run this flag intentionally means "stop here, this smoke test
        # is done," but on --resume it's easy to forget to also pass
        # --max-train-steps 0, which would otherwise silently finalize (and
        # delete the checkpoint for) a long run that's nowhere near done.
        print(
            f"Stopped at --max-train-steps without reaching the full target "
            f"({total_steps_completed}/{total_target_steps} steps done). Checkpoint saved; "
            "re-run with --resume (and --max-train-steps 0, for a real run) to continue."
        )
        _save_checkpoint()
        writer.close()
        return

    writer.close()

    elapsed_seconds = elapsed_seconds_before + (time.perf_counter() - start_time)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # model is wrapped in Opacus's GradSampleModule; save the underlying peft model.
    model._module.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Training finished (either the target step budget was reached, or
    # --max-train-steps intentionally cut it short) rather than being
    # interrupted, so the run is "done" — any resumable checkpoint is stale.
    checkpoint_path.unlink(missing_ok=True)
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
        "private_steps": total_steps_completed,
        "optimizer_steps": total_steps_completed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "bleu": None,
        "rouge_l": None,
        "perplexity": None,
        "train_loss": loss_sum / loss_count if loss_count else None,
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
