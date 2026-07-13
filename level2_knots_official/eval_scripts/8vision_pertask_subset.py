from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from task_merger import get_merge_handler
from utils import evaluate_cliphead, get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed


FINE_TUNED_ACC = {
    "stanford_cars": 74.0,
    "dtd": 58.3,
    "eurosat": 99.0,
    "gtsrb": 92.7,
    "mnist": 99.3,
    "resisc45": 88.4,
    "sun397": 64.5,
    "svhn": 96.2,
}


def _set_dataset_roots(data_root: Path) -> None:
    import dataset.gtsrb as gtsrb
    import dataset.mnist as mnist
    import dataset.svhn as svhn

    mnist.ROOT = str(data_root / "mnist")
    svhn.ROOT = str(data_root / "svhn")
    gtsrb.ROOT = str(data_root / "gtsrb")


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config", "dataset", "accuracy", "normalized_accuracy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official KnOTS Table 1 subset runner.")
    parser.add_argument("--config-name", default="vitB_r16_knots_ties_subset")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-dir", default="results_subset")
    parser.add_argument("--prepare-device", default="cpu")
    parser.add_argument("--eval-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset filter, e.g. --datasets mnist svhn gtsrb.",
    )
    args = parser.parse_args()

    set_seed(420)
    project_root = Path(__file__).resolve().parents[1]
    data_root = (project_root / args.data_root).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    _set_dataset_roots(data_root)

    print(f"Running with config: {args.config_name}")
    print(f"Data root: {data_root}")
    print(f"Prepare device: {args.prepare_device}; eval device: {args.eval_device}")

    raw_config = get_config_from_name(args.config_name, device=args.prepare_device)
    if args.datasets:
        wanted = {name.lower() for name in args.datasets}
        raw_config["dataset"] = [item for item in raw_config["dataset"] if item["name"].lower() in wanted]
        if not raw_config["dataset"]:
            raise ValueError(f"No datasets matched filter: {args.datasets}")
        print("Dataset filter:", ", ".join(item["name"] for item in raw_config["dataset"]))

    all_clip_encodings = [get_clip_encodings(item["clip_encodings"]) for item in raw_config["dataset"]]
    config = prepare_experiment_config(raw_config)

    dataset_names = np.array([item["name"] for item in raw_config["dataset"]])
    dataloaders = np.array([item for item in config["data"]])
    print(raw_config["task_merge_config"])

    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        print("Creating Merge")
        models = np.array([model.cpu() for model in config["models"]["bases"]])
        merge_class = get_merge_handler(config["task_merge_config"]["representation"])
        merge = merge_class(
            deepcopy(models),
            pretrained_model=deepcopy(config["models"]["new"]),
            param_handler=config["param_handler"],
            device=args.prepare_device,
            merge_config=config["task_merge_config"],
        )
        merge.transform(config["task_merge_config"])
        merge.set_scaling_coeffs(config["task_merge_config"]["scaling_coeffs"])
        merged_model = merge.merge(config["task_merge_config"])

        print("Evaluate merged model on selected datasets")
        avg_accuracy = 0.0
        avg_norm_accuracy = 0.0
        for index, loader_dict in enumerate(dataloaders):
            dataset_name = str(dataset_names[index])
            loader = loader_dict["test"]["test"]
            class_vectors = all_clip_encodings[index].to(args.eval_device)
            acc = evaluate_cliphead(merged_model.to(args.eval_device), loader, class_vectors=class_vectors)
            accuracy = float(acc * 100)
            normalized = float((acc * 100) / FINE_TUNED_ACC[dataset_name] * 100)
            print(f"{dataset_name} Normalized accuracy is {np.round(normalized, 3)}")
            print(f"{dataset_name} accuracy is {np.round(accuracy, 3)}")
            rows.append(
                {
                    "config": args.config_name,
                    "dataset": dataset_name,
                    "accuracy": accuracy,
                    "normalized_accuracy": normalized,
                }
            )
            avg_accuracy += accuracy
            avg_norm_accuracy += normalized

        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy /= len(dataloaders)
        print(f"Subset Average Accuracy is {np.round(avg_accuracy, 3)}")
        print(f"Subset Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}")
        rows.append(
            {
                "config": args.config_name,
                "dataset": "subset_average",
                "accuracy": avg_accuracy,
                "normalized_accuracy": avg_norm_accuracy,
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_suffix = "_".join(str(item["name"]) for item in raw_config["dataset"])
    result_stem = f"{args.config_name}_{dataset_suffix}_results"
    _write_csv(rows, results_dir / f"{result_stem}.csv")
    with (results_dir / f"{result_stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(f"Wrote results to {results_dir}")


if __name__ == "__main__":
    main()
