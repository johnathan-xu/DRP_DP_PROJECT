"""Tests for Experiment-1 plot input parsing."""

from pathlib import Path

from src.plot_epsilon_sweep import load_json


def test_load_json_reads_result_file(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text('{"bleu": 1.0}', encoding="utf-8")
    assert load_json(path) == {"bleu": 1.0}
