"""Convert an in-progress train_private.py checkpoint into a loadable adapter.

Use this when a DP-LoRA run was stopped (Ctrl+C, SIGTERM, or you're just
happy with the current checkpoint) before reaching its full step target, and
you want to keep the partially-trained result without resuming training:

    python -m src.checkpoint_to_adapter \\
        --checkpoint artifacts/sequence-length-128/checkpoint.pt \\
        --output-dir artifacts/sequence-length-128

Reads max_seq_len/lora_r/lora_alpha/lora_dropout out of the checkpoint's own
saved args (the same config train_private.py persisted for --resume),
rebuilds the identical base model + tokenizer via build_lora_model, loads
the checkpointed weights onto it, and writes a normal adapter directory that
generate.py can load with --adapter-dir. Does not run any optimizer steps.

Also writes a result JSON (default --output-dir/result.json) with the run
stats already stored in the checkpoint: steps completed, elapsed train time,
noise_multiplier, target epsilon/delta, etc. reported_epsilon and
peak_gpu_memory_mb are left null since those are only ever computed live
against the privacy accountant / CUDA state during an actual training run,
not stored in the checkpoint itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import build_lora_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help="Defaults to --output-dir/result.json.",
    )
    return parser.parse_args()


def _load_module_weights(model, state_dict: dict) -> None:
    """Load a checkpointed module state dict onto a freshly-built model.

    Opacus's GradSampleModule may or may not prefix keys (e.g. "_module.")
    depending on version, so try a direct load first and only fall back to
    stripping a prefix if that fails, rather than depending on a specific
    Opacus internal key-naming convention.
    """
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    stripped = {
        (key[len("_module.") :] if key.startswith("_module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(stripped)


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as error:
        raise ImportError("Install training dependencies with: python -m pip install torch") from error

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]

    model, tokenizer, _ = build_lora_model(
        max_seq_len=saved_args["max_seq_len"],
        lora_r=saved_args["lora_r"],
        lora_alpha=saved_args["lora_alpha"],
        lora_dropout=saved_args["lora_dropout"],
    )
    _load_module_weights(model, checkpoint["module_state_dict"])
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(
        f"Saved adapter ({checkpoint['real_steps_completed']}/"
        f"{checkpoint['total_target_steps']} steps completed) to {args.output_dir}"
    )

    loss_count = checkpoint["loss_count"]
    result = {
        "method": "dp-lora",
        "private": True,
        "complete": checkpoint["real_steps_completed"] >= checkpoint["total_target_steps"],
        "target_epsilon": saved_args["epsilon"],
        "reported_epsilon": None,
        "delta": saved_args["delta"],
        "noise_multiplier": checkpoint["noise_multiplier"],
        "clipping_norm": saved_args["clipping_norm"],
        "seq_len": saved_args["max_seq_len"],
        "trainable_parameters": trainable_parameters,
        "effective_batch_size": saved_args["batch_size"] * saved_args["gradient_accumulation_steps"],
        "physical_batch_size": saved_args["batch_size"],
        "private_steps": checkpoint["real_steps_completed"],
        "optimizer_steps": checkpoint["real_steps_completed"],
        "total_target_steps": checkpoint["total_target_steps"],
        "epochs": saved_args["epochs"],
        "learning_rate": saved_args["learning_rate"],
        "seed": saved_args["seed"],
        "bleu": None,
        "rouge_l": None,
        "perplexity": None,
        "train_loss": checkpoint["loss_sum"] / loss_count if loss_count else None,
        "validation_loss": None,
        "train_seconds": round(checkpoint["elapsed_seconds_before"], 2),
        "peak_gpu_memory_mb": None,
        "adapter_dir": str(args.output_dir),
    }
    result_file = args.result_file or (args.output_dir / "result.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote run stats to {result_file}")


if __name__ == "__main__":
    main()
