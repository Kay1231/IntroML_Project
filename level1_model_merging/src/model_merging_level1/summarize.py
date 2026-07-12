from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from .config import load_project_config


def _metric_value(row: dict[str, Any]) -> float:
    if row["benchmark"] == "gsm8k":
        return float(row.get("accuracy", 0.0))
    if row["benchmark"] == "humaneval":
        return float(row.get("pass_at_1", row.get("syntax_rate", 0.0)))
    return 0.0


def _metric_name(row: dict[str, Any]) -> str:
    if row["benchmark"] == "gsm8k":
        return "accuracy"
    if row["benchmark"] == "humaneval":
        return "pass_at_1" if "pass_at_1" in row else "syntax_rate"
    return "score"


def load_summaries(eval_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metric"] = _metric_name(row)
        row["score"] = _metric_value(row)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "benchmark", "metric", "score", "num_examples", "details_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "benchmark": row["benchmark"],
                    "metric": row["metric"],
                    "score": f"{row['score']:.6f}",
                    "num_examples": row["num_examples"],
                    "details_path": row["details_path"],
                }
            )


def _load_plotting():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return plt, np, pd


def plot_bar(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt, _, pd = _load_plotting()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    pivot = frame.pivot_table(index="model", columns="benchmark", values="score", aggfunc="first").fillna(0.0)
    axis = pivot.plot(kind="bar", figsize=(11, 5), ylim=(0, 1), rot=35)
    axis.set_ylabel("Score")
    axis.set_title("Level 1 Benchmark Scores")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Benchmark")
    plt.tight_layout()
    plt.savefig(output_dir / "bar_accuracy.png", dpi=180)
    plt.close()


def plot_radar(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt, np, pd = _load_plotting()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    pivot = frame.pivot_table(index="model", columns="benchmark", values="score", aggfunc="first").fillna(0.0)
    labels = list(pivot.columns)
    if len(labels) < 2:
        return

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    axis = fig.add_subplot(111, polar=True)
    for model_name, values in pivot.iterrows():
        score_values = values.tolist()
        score_values += score_values[:1]
        axis.plot(angles, score_values, linewidth=1.8, label=model_name)
        axis.fill(angles, score_values, alpha=0.08)

    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_title("Capability Retention Radar")
    axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "radar_accuracy.png", dpi=180)
    plt.close()


def _extract_math_weight(model_name: str) -> float | None:
    match = re.search(r"code(\d{3})_math(\d{3})", model_name)
    if not match:
        return None
    return int(match.group(2)) / 100.0


def plot_slerp_curve(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt, _, pd = _load_plotting()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame = frame[frame["model"].str.startswith("slerp_")].copy()
    if frame.empty:
        return
    frame["math_weight"] = frame["model"].map(_extract_math_weight)
    frame = frame.dropna(subset=["math_weight"])
    if frame.empty:
        return

    fig, axis = plt.subplots(figsize=(8, 5))
    for benchmark, group in frame.groupby("benchmark"):
        group = group.sort_values("math_weight")
        axis.plot(group["math_weight"], group["score"], marker="o", label=benchmark)
    axis.set_xlabel("Math expert weight in SLERP")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_title("SLERP Weight Ratio Curve")
    axis.grid(alpha=0.25)
    axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "slerp_weight_curve.png", dpi=180)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Level 1 evaluation results.")
    parser.add_argument("--config", default="configs/default_experiment.yaml", help="Path to the experiment YAML.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_project_config(args.config)
    config.plots_dir().mkdir(parents=True, exist_ok=True)
    rows = load_summaries(config.eval_dir())
    if not rows:
        raise SystemExit(f"No summary files found in {config.eval_dir()}")

    csv_path = config.results_dir / "results_summary.csv"
    write_csv(rows, csv_path)
    plot_bar(rows, config.plots_dir())
    plot_radar(rows, config.plots_dir())
    plot_slerp_curve(rows, config.plots_dir())
    print(f"Wrote {csv_path}")
    print(f"Wrote plots to {config.plots_dir()}")


if __name__ == "__main__":
    main()
