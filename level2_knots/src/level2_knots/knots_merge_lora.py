from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


def _load_runtime():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, load_file, AutoModelForCausalLM, AutoTokenizer


def _torch_dtype(torch_module: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    if not hasattr(torch_module, value):
        raise ValueError(f"Unknown torch dtype: {value}")
    return getattr(torch_module, value)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def _adapter_model_path(adapter_dir: Path) -> Path:
    path = adapter_dir / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter weights: {path}")
    return path


def _load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    path = adapter_dir / "adapter_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_adapter_configs(
    *,
    adapter_dirs: dict[str, Path],
    adapter_configs: dict[str, dict[str, Any]],
    base_model: str,
) -> None:
    for name, adapter_dir in adapter_dirs.items():
        declared_base = str(adapter_configs[name].get("base_model_name_or_path"))
        if declared_base != base_model:
            raise ValueError(
                f"Adapter {name} at {adapter_dir} declares base {declared_base!r}, "
                f"but config base_model is {base_model!r}."
            )
        if adapter_configs[name].get("peft_type") != "LORA":
            raise ValueError(f"Adapter {name} is not a LoRA adapter.")

    ranks = {name: int(config.get("r", 0)) for name, config in adapter_configs.items()}
    alphas = {name: int(config.get("lora_alpha", 0)) for name, config in adapter_configs.items()}
    if any(rank <= 0 for rank in ranks.values()):
        raise ValueError(f"Invalid LoRA ranks: {ranks}")
    if any(alpha <= 0 for alpha in alphas.values()):
        raise ValueError(f"Invalid LoRA alphas: {alphas}")


def _extract_lora_pairs(adapter_state: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    pairs: dict[str, tuple[Any, Any]] = {}
    for key in adapter_state:
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        a_key = f"{prefix}.lora_A.weight"
        b_key = f"{prefix}.lora_B.weight"
        if b_key not in adapter_state:
            raise KeyError(f"Missing matching lora_B weight for {a_key}")
        base_key = prefix
        if base_key.startswith("base_model.model."):
            base_key = base_key[len("base_model.model.") :]
        weight_key = f"{base_key}.weight"
        pairs[weight_key] = (adapter_state[a_key], adapter_state[b_key])
    return pairs


def _make_block_diag(torch: Any, matrices: list[Any]) -> Any:
    rows = sum(matrix.shape[0] for matrix in matrices)
    cols = sum(matrix.shape[1] for matrix in matrices)
    result = torch.zeros((rows, cols), dtype=torch.float32, device="cpu")
    row_offset = 0
    col_offset = 0
    for matrix in matrices:
        matrix = matrix.to(dtype=torch.float32, device="cpu")
        row_count, col_count = matrix.shape
        result[row_offset : row_offset + row_count, col_offset : col_offset + col_count] = matrix
        row_offset += row_count
        col_offset += col_count
    return result


def _trim_to_density(torch: Any, values: Any, density: float) -> Any:
    if density >= 1.0:
        return torch.ones_like(values, dtype=torch.bool)
    if density <= 0.0:
        return torch.zeros_like(values, dtype=torch.bool)

    flat_abs = values.abs().reshape(-1)
    keep = max(1, math.ceil(flat_abs.numel() * density))
    if keep >= flat_abs.numel():
        return torch.ones_like(values, dtype=torch.bool)
    threshold = torch.topk(flat_abs, keep, sorted=False).values.min()
    return values.abs() >= threshold


def _merge_task_arithmetic(torch: Any, aligned_updates: list[Any], weights: list[float]) -> Any:
    merged = torch.zeros_like(aligned_updates[0])
    for aligned, weight in zip(aligned_updates, weights):
        merged = merged + aligned * float(weight)
    return merged


def _merge_ties(torch: Any, aligned_updates: list[Any], weights: list[float], density: float) -> Any:
    weighted = torch.stack(
        [aligned * float(weight) for aligned, weight in zip(aligned_updates, weights)],
        dim=0,
    )
    masks = torch.stack([_trim_to_density(torch, value, density) for value in weighted], dim=0)
    trimmed = weighted * masks

    elected_sign = torch.sign(trimmed.sum(dim=0))
    fallback_sign = torch.sign(weighted.sum(dim=0))
    elected_sign = torch.where(elected_sign == 0, fallback_sign, elected_sign)

    agrees = (torch.sign(trimmed) == elected_sign.unsqueeze(0)) & masks & (trimmed != 0)
    contributors = agrees.sum(dim=0)
    agreed_sum = torch.where(agrees, trimmed, torch.zeros_like(trimmed)).sum(dim=0)
    return torch.where(contributors > 0, agreed_sum / contributors.clamp_min(1), torch.zeros_like(agreed_sum))


def _knots_merge_delta(
    *,
    torch: Any,
    lora_pairs: list[tuple[Any, Any]],
    scales: list[float],
    weights: list[float],
    method: str,
    density: float,
) -> Any:
    # LoRA delta for a task is B @ A * alpha/r.  KnOTS first represents every
    # task update in a shared SVD coordinate system, then merges in that space.
    left_factors = []
    right_factors = []
    for (a_weight, b_weight), scale in zip(lora_pairs, scales):
        left_factors.append(b_weight.to(dtype=torch.float32, device="cpu") * float(scale))
        right_factors.append(a_weight.to(dtype=torch.float32, device="cpu"))

    left = torch.cat(left_factors, dim=1)
    right = _make_block_diag(torch, right_factors)

    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T.contiguous(), mode="reduced")
    small_matrix = r_left @ r_right.T
    u_small, singular_values, vh_small = torch.linalg.svd(small_matrix, full_matrices=False)

    left_basis = q_left @ u_small
    vh = vh_small @ q_right.T

    aligned_updates = []
    col_offset = 0
    for right_factor in right_factors:
        col_count = right_factor.shape[1]
        aligned_updates.append(singular_values.unsqueeze(1) * vh[:, col_offset : col_offset + col_count])
        col_offset += col_count

    if method == "task_arithmetic":
        merged_aligned = _merge_task_arithmetic(torch, aligned_updates, weights)
    elif method == "ties":
        merged_aligned = _merge_ties(torch, aligned_updates, weights, density)
    else:
        raise ValueError(f"Unsupported KnOTS merge method: {method}")

    return left_basis @ merged_aligned


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KnOTS-style SVD-aligned merge for two same-base LoRA adapters.")
    parser.add_argument("--config", default="configs/knots_qwen_lora_v2.yaml", help="Level 2 experiment YAML.")
    parser.add_argument("--name", required=True, help="Output model name under paths.merge_output_dir.")
    parser.add_argument("--code-weight", type=float, required=True)
    parser.add_argument("--math-weight", type=float, required=True)
    parser.add_argument("--method", choices=["task_arithmetic", "ties"], default="ties")
    parser.add_argument("--density", type=float, default=0.8, help="TIES trim density in KnOTS coordinates.")
    parser.add_argument("--torch-dtype", default="bfloat16", help="auto, bfloat16, float16, or float32.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_path = Path(args.config).resolve()
    raw = _load_yaml(config_path)
    project_root = config_path.parent.parent

    base_model = str(raw["base_model"])
    paths = raw.get("paths", {})
    output_root = _resolve_path(project_root, paths.get("merge_output_dir", "merged_models"))
    hf_cache_dir = _resolve_path(project_root, paths["hf_cache_dir"]) if paths.get("hf_cache_dir") else None
    if hf_cache_dir:
        # Some transformers adapter-probing paths consult HF_* env vars before
        # honoring cache_dir, so pin them here to keep all artifacts under D:.
        os.environ["HF_HUB_CACHE"] = str(hf_cache_dir)
        os.environ["HF_HOME"] = str(project_root.parent / "hf_home")
        os.environ["HF_DATASETS_CACHE"] = str(project_root.parent / "hf_datasets_cache")
    output_dir = output_root / args.name

    if output_dir.exists() and not args.force:
        print(f"Skip {args.name}: {output_dir} already exists. Use --force to overwrite.")
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)

    adapter_refs = dict(raw["lora_adapters"])
    adapter_dirs = {
        "code": _resolve_path(project_root, str(adapter_refs["code"])),
        "math": _resolve_path(project_root, str(adapter_refs["math"])),
    }
    adapter_configs = {name: _load_adapter_config(path) for name, path in adapter_dirs.items()}
    _validate_adapter_configs(adapter_dirs=adapter_dirs, adapter_configs=adapter_configs, base_model=base_model)

    torch, load_file, AutoModelForCausalLM, AutoTokenizer = _load_runtime()
    adapter_states = {
        name: load_file(str(_adapter_model_path(path)), device="cpu") for name, path in adapter_dirs.items()
    }
    lora_pairs_by_task = {name: _extract_lora_pairs(state) for name, state in adapter_states.items()}
    common_weight_keys = sorted(set(lora_pairs_by_task["code"]).intersection(lora_pairs_by_task["math"]))
    if not common_weight_keys:
        raise ValueError("No common LoRA target weights found between code and math adapters.")

    missing_code = sorted(set(lora_pairs_by_task["math"]).difference(lora_pairs_by_task["code"]))
    missing_math = sorted(set(lora_pairs_by_task["code"]).difference(lora_pairs_by_task["math"]))
    if missing_code or missing_math:
        raise ValueError(f"Adapters target different weights. Missing code={missing_code}, missing math={missing_math}")

    scales = [
        float(adapter_configs["code"]["lora_alpha"]) / float(adapter_configs["code"]["r"]),
        float(adapter_configs["math"]["lora_alpha"]) / float(adapter_configs["math"]["r"]),
    ]
    weights = [float(args.code_weight), float(args.math_weight)]

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=_torch_dtype(torch, args.torch_dtype),
        device_map=None,
        low_cpu_mem_usage=True,
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
    )
    named_parameters = dict(model.named_parameters())

    print(
        "Applying KnOTS merge:",
        f"name={args.name}",
        f"method={args.method}",
        f"weights=code:{weights[0]},math:{weights[1]}",
        f"density={args.density}",
        f"layers={len(common_weight_keys)}",
    )
    with torch.no_grad():
        for index, weight_key in enumerate(common_weight_keys, start=1):
            if weight_key not in named_parameters:
                raise KeyError(f"Base model parameter not found: {weight_key}")
            merged_delta = _knots_merge_delta(
                torch=torch,
                lora_pairs=[
                    lora_pairs_by_task["code"][weight_key],
                    lora_pairs_by_task["math"][weight_key],
                ],
                scales=scales,
                weights=weights,
                method=args.method,
                density=float(args.density),
            )
            parameter = named_parameters[weight_key]
            if tuple(parameter.shape) != tuple(merged_delta.shape):
                raise ValueError(
                    f"Shape mismatch for {weight_key}: base {tuple(parameter.shape)} vs delta {tuple(merged_delta.shape)}"
                )
            parameter.add_(merged_delta.to(dtype=parameter.dtype))
            print(f"[{index:03d}/{len(common_weight_keys):03d}] merged {weight_key}")
            del merged_delta

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=str(hf_cache_dir) if hf_cache_dir else None)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "paper": "Model merging with SVD to tie the knots (ICLR 2025)",
        "paper_url": "https://arxiv.org/abs/2410.19735",
        "official_code": "https://github.com/gstoica27/KnOTS",
        "base_model": base_model,
        "adapters": {name: str(path) for name, path in adapter_dirs.items()},
        "weights": {"code": weights[0], "math": weights[1]},
        "method": args.method,
        "density": args.density,
        "implementation_note": (
            "For each LoRA target weight, this script forms a compact SVD of the concatenated full "
            "task updates [B_code A_code, B_math A_math], merges task coordinates with task arithmetic "
            "or TIES, and writes the merged delta directly into the base model weights."
        ),
    }
    with (output_dir / "knots_merge_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Saved KnOTS merged model to {output_dir}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
