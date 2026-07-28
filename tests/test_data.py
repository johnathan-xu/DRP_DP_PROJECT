"""Tests for the shared E2E preprocessing contract; no ML packages required."""

import re

from src.data import IGNORE_INDEX, MR_TOKEN, SEP_TOKEN, configure_tokenizer, preprocess_example


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token = None
    padding_side = None

    def __init__(self):
        self.vocab = {self.eos_token: 0}
        self.added_special_tokens = []

    def add_special_tokens(self, values):
        self.added_special_tokens.extend(values["additional_special_tokens"])
        for token in values["additional_special_tokens"]:
            self.vocab.setdefault(token, len(self.vocab))
        return len(values["additional_special_tokens"])

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"<[^>]+>|[^\s<]+", text))
        tokens = [match.group() for match in matches]
        offsets = [match.span() for match in matches]
        ids = [self.vocab.setdefault(token, len(self.vocab)) for token in tokens]
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        attention_mask = [1] * len(ids)
        if kwargs.get("padding") == "max_length" and max_length is not None:
            padding = max_length - len(ids)
            ids += [self.vocab[self.eos_token]] * padding
            attention_mask += [0] * padding
            offsets += [(0, 0)] * padding
        result = {"input_ids": ids, "attention_mask": attention_mask}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


def test_preprocess_masks_prompt_and_padding():
    tokenizer = FakeTokenizer()
    configure_tokenizer(tokenizer)
    record = {
        "meaning_representation": "name[The Eagle], food[French]",
        "target": "The Eagle serves French food.",
    }
    result = preprocess_example(record, tokenizer, max_seq_len=20)
    prompt_length = len(tokenizer(f"{MR_TOKEN} {record['meaning_representation']} {SEP_TOKEN} ")["input_ids"])

    assert tokenizer.added_special_tokens == [MR_TOKEN, SEP_TOKEN]
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "right"
    assert result["labels"][:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert any(label != IGNORE_INDEX for label in result["labels"])
    assert result["labels"][-1] == IGNORE_INDEX
    assert len(result["input_ids"]) == len(result["attention_mask"]) == len(result["labels"]) == 20
