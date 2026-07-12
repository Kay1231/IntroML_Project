from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    project_name: str
    base_model: str
    expert_models: dict[str, str]
    merge_dtype: str
    merge_config_dir: Path
    merge_output_dir: Path
    results_dir: Path
    hf_cache_dir: str | None
    methods: list[dict[str, Any]]
    eval: dict[str, Any]

    def merged_model_dir(self, method_name: str) -> Path:
        return self.merge_output_dir / method_name

    def eval_dir(self) -> Path:
        return self.results_dir / "eval"

    def plots_dir(self) -> Path:
        return self.results_dir / "plots"


def _resolve_path(project_root: Path, value: str | None, default: str) -> Path:
    raw = value or default
    path = Path(raw)
    return (path if path.is_absolute() else project_root / path).resolve()


def load_project_config(config_path: str | Path) -> ProjectConfig:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    project_root = path.parent.parent
    paths = raw.get("paths", {})
    hf_cache_dir = paths.get("hf_cache_dir")
    resolved_hf_cache_dir = (
        str(_resolve_path(project_root, hf_cache_dir, "hf_cache")) if hf_cache_dir else None
    )

    return ProjectConfig(
        project_root=project_root,
        project_name=raw["project_name"],
        base_model=raw["base_model"],
        expert_models=dict(raw["expert_models"]),
        merge_dtype=raw.get("merge_dtype", "bfloat16"),
        merge_config_dir=_resolve_path(project_root, paths.get("merge_config_dir"), "generated_merge_configs"),
        merge_output_dir=_resolve_path(project_root, paths.get("merge_output_dir"), "merged_models"),
        results_dir=_resolve_path(project_root, paths.get("results_dir"), "results"),
        hf_cache_dir=resolved_hf_cache_dir,
        methods=list(raw["methods"]),
        eval=dict(raw.get("eval", {})),
    )


def method_names(config: ProjectConfig) -> list[str]:
    return [str(method["name"]) for method in config.methods]


def select_methods(config: ProjectConfig, only: list[str] | None = None) -> list[dict[str, Any]]:
    if not only:
        return config.methods

    allowed = set(only)
    selected = [method for method in config.methods if method["name"] in allowed]
    missing = sorted(allowed.difference(method["name"] for method in selected))
    if missing:
        available = ", ".join(method_names(config))
        raise ValueError(f"Unknown method(s): {missing}. Available methods: {available}")
    return selected
