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
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import build_lora_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(
        f"Saved adapter ({checkpoint['real_steps_completed']}/"
        f"{checkpoint['total_target_steps']} steps completed) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
