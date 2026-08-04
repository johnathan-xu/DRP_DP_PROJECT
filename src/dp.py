"""Opacus helpers for private causal-LM fine-tuning.

The helpers here deliberately keep the DP-specific compatibility details out of
the ordinary LoRA training loop. They are for research experiments only; the
result JSON records that secure RNG is disabled by default.
"""

from __future__ import annotations

from typing import Any


def register_gpt2_grad_sampler() -> None:
    """Register dp-transformers' per-sample sampler for GPT-2 ``Conv1D``.

    dp-transformers currently imports ``Conv1D`` from its older Transformers
    location. Newer Transformers moved it to ``pytorch_utils``, so expose a
    backwards-compatible alias before importing the sampler registration code.
    This must run before Opacus makes the model private.
    """
    import transformers.modeling_utils

    if not hasattr(transformers.modeling_utils, "Conv1D"):
        from transformers.pytorch_utils import Conv1D

        transformers.modeling_utils.Conv1D = Conv1D
    import dp_transformers.grad_sample.transformers.conv_1d  # noqa: F401


class PrivateCausalLMDataset:
    """Expose Hugging Face rows as tensors that Opacus can inspect safely."""

    def __init__(self, dataset: Any):
        self.dataset = dataset.with_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        return row["input_ids"], row["attention_mask"], row["labels"]


def private_causal_lm_collate(rows: list[Any]) -> dict[str, Any]:
    """Stack causal-LM examples and create batch-shaped GPT-2 position IDs."""
    import torch

    input_ids, attention_mask, labels = (torch.stack(items) for items in zip(*rows))
    sequence_length = input_ids.shape[1]
    position_ids = torch.arange(sequence_length, dtype=torch.long).repeat(
        input_ids.shape[0], 1
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": position_ids,
    }


def make_private_with_budget(
    *,
    model: Any,
    optimizer: Any,
    dataset: Any,
    batch_size: int,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float,
    accountant: str = "prv",
):
    """Attach Opacus using Poisson sampling and calibrate noise from a budget.

    ``wrap_model=False`` keeps Hugging Face/PEFT's model wrapper intact; Opacus
    attaches hooks directly and returns an object that must be cleaned up after
    training. ``rand_on_empty=True`` lets rare empty Poisson draws consume a
    private step without attempting to train on an empty tensor.
    """
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader

    if target_epsilon <= 0 or target_delta <= 0 or target_delta >= 1:
        raise ValueError("target_epsilon must be positive and target_delta must be in (0, 1)")
    private_dataset = PrivateCausalLMDataset(dataset)
    data_loader = DataLoader(
        private_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=private_causal_lm_collate,
    )
    privacy_engine = PrivacyEngine(accountant=accountant, secure_mode=False)
    hooks, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,
        rand_on_empty=True,
        wrap_model=False,
    )
    # Opacus 1.6 derives empty-batch metadata by iterating dataset[0]. A Hugging
    # Face example is a mapping, so prime the wrapped collator with one valid
    # batch to preserve our named batch structure for a rare first empty draw.
    private_loader.collate_fn.first_batch = private_causal_lm_collate([private_dataset[0]])
    return privacy_engine, hooks, private_optimizer, private_loader
