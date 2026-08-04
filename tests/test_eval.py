"""Tests for metric input validation and shared scoring."""

import pytest

from src.eval import calculate_metrics, rouge_l_f1


def test_calculate_metrics_returns_expected_scores():
    metrics = calculate_metrics(["The Eagle is French."], ["The Eagle is French."])
    assert metrics["bleu"] == 100.0
    assert metrics["rouge_l"] == 1.0


def test_calculate_metrics_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        calculate_metrics(["prediction"], [])


def test_rouge_l_f1_handles_empty_prediction():
    assert rouge_l_f1("", "reference text") == 0.0
