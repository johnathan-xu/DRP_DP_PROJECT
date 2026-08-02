"""Shared E2E NLG preprocessing for conditional DistilGPT2 fine-tuning."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any


MODEL_NAME = "distilbert/distilgpt2"
DATASET_NAME = "GEM/e2e_nlg"
OFFICIAL_SPLITS = ("train", "validation", "test")
MR_TOKEN = "<MR>"
SEP_TOKEN = "<SEP>"
IGNORE_INDEX = -100


def configure_tokenizer(tokenizer: Any) -> int:
    """Apply shared special-token and padding settings.

    Return the number of tokens added. The training script must subsequently
    call ``model.resize_token_embeddings(len(tokenizer))``.
    """
    if tokenizer.eos_token is None:
        raise ValueError("The selected tokenizer must define an EOS token.")
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [MR_TOKEN, SEP_TOKEN]}
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return added


def format_prompt(meaning_representation: str) -> str:
    """Create the conditioning prefix; its tokens receive no loss."""
    if not isinstance(meaning_representation, str) or not meaning_representation.strip():
        raise ValueError("meaning_representation must be a non-empty string")
    return f"{MR_TOKEN} {meaning_representation.strip()} {SEP_TOKEN} "


def format_example(meaning_representation: str, target: str, eos_token: str) -> str:
    """Create the canonical full causal-LM input for one E2E record."""
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty string")
    return f"{format_prompt(meaning_representation)}{target.strip()}{eos_token}"


def _get_text(example: Mapping[str, Any], field: str) -> str:
    value = example.get(field)
    if not isinstance(value, str):
        raise KeyError(f"Expected string field {field!r}; found {sorted(example)}")
    return value


def preprocess_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_seq_len: int,
    mr_field: str = "meaning_representation",
    target_field: str = "target",
) -> dict[str, list[int]]:
    """Tokenize an E2E item, masking prompt and padding labels with ``-100``."""
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    mr = _get_text(example, mr_field)
    target = _get_text(example, target_field)
    prompt = format_prompt(mr)
    full_text = format_example(mr, target, tokenizer.eos_token)
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    offsets = list(encoded["offset_mapping"])
    prompt_boundary = len(prompt)

    # GPT-2 commonly merges the space after <SEP> with the first target word.
    # Character offsets preserve the intended loss boundary where separately
    # tokenizing the prompt and the full input would not.
    labels = [
        token_id if attention and end > prompt_boundary else IGNORE_INDEX
        for token_id, attention, (_, end) in zip(
            input_ids, attention_mask, offsets, strict=True
        )
    ]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def has_supervised_tokens(example: Mapping[str, Any]) -> bool:
    """Discard an item if truncation removed every target token."""
    return any(label != IGNORE_INDEX for label in example["labels"])


def load_e2e_splits(*, max_seq_len: int = 128, dataset_name: str = DATASET_NAME):
    """Load, tokenize, and filter all official E2E splits consistently."""
    try:
        from datasets import DatasetDict, load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError("Install datasets and transformers before loading E2E.") from error

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    configure_tokenizer(tokenizer)
    raw_splits = load_dataset(dataset_name, trust_remote_code=True)
    missing_splits = set(OFFICIAL_SPLITS).difference(raw_splits)
    if missing_splits:
        raise ValueError(f"E2E dataset is missing expected splits: {sorted(missing_splits)}")
    raw_splits = DatasetDict({name: raw_splits[name] for name in OFFICIAL_SPLITS})
    first_split = next(iter(raw_splits))
    processed_splits = raw_splits.map(
        lambda row: preprocess_example(row, tokenizer, max_seq_len=max_seq_len),
        remove_columns=raw_splits[first_split].column_names,
        desc="Tokenizing E2E NLG",
    ).filter(has_supervised_tokens, desc="Dropping examples without target tokens")
    return processed_splits, tokenizer


def build_lora_model(
    *, max_seq_len: int, lora_r: int, lora_alpha: int, lora_dropout: float
):
    """Load E2E splits/tokenizer and return a LoRA-wrapped DistilGPT2 model.

    Both the private and non-private training scripts call this so they share
    identical tokenizer/embedding/LoRA configuration. Returns
    ``(model, tokenizer, splits)``.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise ImportError(
            "Install training dependencies with: python -m pip install peft accelerate"
        ) from error

    splits, tokenizer = load_e2e_splits(max_seq_len=max_seq_len)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # The data module adds <MR> and <SEP>, so the model must expose matching
    # embedding rows before the adapter is attached and training begins.
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["c_attn", "c_proj"],
        # GPT-2 uses its Conv1D projection wrapper rather than nn.Linear.
        fan_in_fan_out=True,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, splits


def main() -> None:
    """Print split sizes and one decoded input/target for a manual smoke test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()
    splits, tokenizer = load_e2e_splits(max_seq_len=args.max_seq_len)
    print("Splits:", {name: len(split) for name, split in splits.items()})
    example = splits[next(iter(splits))][0]
    target_ids = [
        token_id
        for token_id, label in zip(example["input_ids"], example["labels"])
        if label != IGNORE_INDEX
    ]
    active_input_ids = [
        token_id
        for token_id, attention in zip(example["input_ids"], example["attention_mask"])
        if attention
    ]
    print("Input:", tokenizer.decode(active_input_ids, skip_special_tokens=False))
    print("Supervised target:", tokenizer.decode(target_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
