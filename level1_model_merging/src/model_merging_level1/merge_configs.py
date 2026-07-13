from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .config import ProjectConfig, load_project_config, select_methods


def _tokenizer_block() -> dict[str, Any]:
    return {"source": "union"}


def _slerp_config(config: ProjectConfig, method: dict[str, Any]) -> dict[str, Any]:
    code_model = config.expert_models["code"]
    math_model = config.expert_models["math"]
    parameters = dict(method.get("parameters", {}))

    return {
        "models": [
            {"model": code_model},
            {"model": math_model},
        ],
        "merge_method": "slerp",
        "base_model": code_model,
        "parameters": {
            "t": float(parameters.get("t", 0.5)),
        },
        "dtype": config.merge_dtype,
        "tokenizer": _tokenizer_block(),
        "chat_template": "auto",
    }


def _weighted_model_block(model_id: str, weight: float, density: float | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"weight": float(weight)}
    if density is not None:
        params["density"] = float(density)
    return {"model": model_id, "parameters": params}


def _task_or_ties_config(config: ProjectConfig, method: dict[str, Any]) -> dict[str, Any]:
    merge_method = str(method["merge_method"])
    weights = dict(method.get("weights", {}))
    parameters = dict(method.get("parameters", {}))
    density = parameters.pop("density", None)

    code_weight = float(weights.get("code", 0.5))
    math_weight = float(weights.get("math", 0.5))

    return {
        "models": [
            _weighted_model_block(config.expert_models["code"], code_weight, density),
            _weighted_model_block(config.expert_models["math"], math_weight, density),
        ],
        "merge_method": merge_method,
        "base_model": config.base_model,
        "parameters": parameters,
        "dtype": config.merge_dtype,
        "tokenizer": _tokenizer_block(),
        "chat_template": "auto",
    }


def _linear_config(config: ProjectConfig, method: dict[str, Any]) -> dict[str, Any]:
    weights = dict(method.get("weights", {}))
    parameters = dict(method.get("parameters", {}))

    code_weight = float(weights.get("code", 0.5))
    math_weight = float(weights.get("math", 0.5))

    return {
        "models": [
            _weighted_model_block(config.expert_models["code"], code_weight),
            _weighted_model_block(config.expert_models["math"], math_weight),
        ],
        "merge_method": "linear",
        "parameters": {
            "normalize": bool(parameters.get("normalize", True)),
        },
        "dtype": config.merge_dtype,
        "tokenizer": _tokenizer_block(),
        "chat_template": "auto",
    }


def render_merge_config(config: ProjectConfig, method: dict[str, Any]) -> dict[str, Any]:
    merge_method = str(method["merge_method"])
    if merge_method == "slerp":
        return _slerp_config(config, method)
    if merge_method == "linear":
        return _linear_config(config, method)
    if merge_method in {
        "task_arithmetic",
        "ties",
        "dare_ties",
        "dare_linear",
        "breadcrumbs",
        "breadcrumbs_ties",
        "della",
        "della_linear",
    }:
        return _task_or_ties_config(config, method)
    raise ValueError(f"Unsupported merge method: {merge_method}")


def write_merge_configs(config: ProjectConfig, only: list[str] | None = None) -> list[Path]:
    config.merge_config_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for method in select_methods(config, only):
        merge_config = render_merge_config(config, method)
        path = config.merge_config_dir / f"{method['name']}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(merge_config, handle, sort_keys=False, allow_unicode=True)
        written.append(path)

    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MergeKit YAML files for Level 1 experiments.")
    parser.add_argument("--config", default="configs/default_experiment.yaml", help="Path to the experiment YAML.")
    parser.add_argument("--only", action="append", default=None, help="Generate only this method. May be repeated.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_project_config(args.config)
    written = write_merge_configs(config, args.only)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
