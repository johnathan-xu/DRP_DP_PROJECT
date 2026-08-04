"""Create the Experiment-1 privacy--utility trade-off figure.

Example:
    python -m src.plot_epsilon_sweep \
      --baseline results/baseline_evaluation.json \
      --dp-results results/evaluation_eps*.json \
      --output plots/epsilon_sweep.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DP-LoRA BLEU/PPL against epsilon.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dp-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("plots/epsilon_sweep.png"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    rows = [load_json(path) for path in args.dp_results]
    rows.sort(key=lambda row: float(row["target_epsilon"]))
    expected = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    epsilons = [float(row["target_epsilon"]) for row in rows]
    if epsilons != expected:
        raise ValueError(f"Expected exactly {expected}; received {epsilons}")
    for metric in ("bleu", "perplexity"):
        if baseline.get(metric) is None or any(row.get(metric) is None for row in rows):
            raise ValueError(f"Every result must contain {metric}")

    import matplotlib.pyplot as plt

    figure, (bleu_axis, ppl_axis) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    bleu_axis.plot(epsilons, [row["bleu"] for row in rows], marker="o", label="DP-LoRA")
    bleu_axis.axhline(baseline["bleu"], color="black", linestyle="--", label="Non-private LoRA")
    bleu_axis.set(xscale="log", xlabel="Privacy budget ε (larger = weaker privacy)", ylabel="BLEU")
    bleu_axis.legend()

    ppl_axis.plot(epsilons, [row["perplexity"] for row in rows], marker="o", label="DP-LoRA")
    ppl_axis.axhline(baseline["perplexity"], color="black", linestyle="--", label="Non-private LoRA")
    ppl_axis.set(xscale="log", xlabel="Privacy budget ε (larger = weaker privacy)", ylabel="Perplexity")
    ppl_axis.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
