from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .config import load_project_config


def _load_runtime():
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, PeftConfig, PeftModel, AutoModelForCausalLM, AutoTokenizer


def _torch_dtype(torch_module: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    if not hasattr(torch_module, value):
        raise ValueError(f"Unknown torch dtype: {value}")
    return getattr(torch_module, value)


def _load_raw_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_adapter_ref(project_root: Path, adapter_ref: str) -> str:
    path = Path(adapter_ref)
    if path.is_absolute():
        return str(path)
    local_path = project_root / path
    return str(local_path) if local_path.exists() else adapter_ref


def _validate_adapter_base(
    *,
    alias: str,
    adapter_ref: str,
    expected_base: str,
    hf_cache_dir: str | None,
    peft_config_cls: Any,
) -> None:
    peft_config = peft_config_cls.from_pretrained(adapter_ref, cache_dir=hf_cache_dir)
    declared_base = str(peft_config.base_model_name_or_path)
    if declared_base != expected_base:
        raise ValueError(
            f"{alias} adapter declares base {declared_base!r}, "
            f"but experiment base is {expected_base!r}."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge same-base LoRA adapters and materialize the result.")
    parser.add_argument("--config", default="configs/lora_experiment.yaml", help="Path to LoRA experiment YAML.")
    parser.add_argument("--name", required=True, help="Output model name under the configured merge_output_dir.")
    parser.add_argument("--code-weight", type=float, required=True, help="Weight for the code LoRA adapter.")
    parser.add_argument("--math-weight", type=float, required=True, help="Weight for the math LoRA adapter.")
    parser.add_argument(
        "--combination-type",
        default="svd",
        help="PEFT adapter merge type: svd, ties_svd, dare_linear_svd, dare_ties_svd, etc.",
    )
    parser.add_argument("--svd-rank", type=int, default=64, help="Rank for SVD-based merged adapter.")
    parser.add_argument("--density", type=float, default=None, help="Density for pruning/TIES-style merges.")
    parser.add_argument("--torch-dtype", default=None, help="auto, bfloat16, float16, or float32.")
    parser.add_argument("--device-map", default=None, help="auto, cuda, cpu, or none.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_project_config(args.config)
    raw = _load_raw_config(args.config)
    adapters = dict(raw.get("lora_adapters", {}))
    if {"code", "math"}.difference(adapters):
        raise ValueError("lora_adapters must contain both 'code' and 'math'.")
    adapters = {
        alias: _resolve_adapter_ref(config.project_root, str(adapter_ref))
        for alias, adapter_ref in adapters.items()
    }

    output_dir = config.merged_model_dir(args.name)
    if output_dir.exists() and not args.force:
        print(f"Skip {args.name}: {output_dir} already exists. Use --force to overwrite.")
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)

    torch, PeftConfig, PeftModel, AutoModelForCausalLM, AutoTokenizer = _load_runtime()
    for alias in ("code", "math"):
        _validate_adapter_base(
            alias=alias,
            adapter_ref=str(adapters[alias]),
            expected_base=config.base_model,
            hf_cache_dir=config.hf_cache_dir,
            peft_config_cls=PeftConfig,
        )

    gen_config = config.eval.get("generation", {})
    dtype_name = args.torch_dtype or gen_config.get("torch_dtype", "auto")
    device_map = args.device_map or gen_config.get("device_map", "auto")
    device_map_value = None if device_map == "none" else device_map

    print(f"Loading base: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=_torch_dtype(torch, dtype_name),
        device_map=device_map_value,
        low_cpu_mem_usage=True,
        cache_dir=config.hf_cache_dir,
    )

    print(f"Loading code adapter: {adapters['code']}")
    model = PeftModel.from_pretrained(
        model,
        str(adapters["code"]),
        adapter_name="code",
        cache_dir=config.hf_cache_dir,
    )
    print(f"Loading math adapter: {adapters['math']}")
    model.load_adapter(
        str(adapters["math"]),
        adapter_name="math",
        cache_dir=config.hf_cache_dir,
    )

    print(
        "Merging adapters:",
        f"type={args.combination_type}",
        f"code={args.code_weight}",
        f"math={args.math_weight}",
        f"svd_rank={args.svd_rank}",
        f"density={args.density}",
    )
    model.base_model.add_weighted_adapter(
        adapters=["code", "math"],
        weights=[args.code_weight, args.math_weight],
        adapter_name="merged",
        combination_type=args.combination_type,
        svd_rank=args.svd_rank,
        svd_full_matrices=False,
        density=args.density,
    )
    model.set_adapter("merged")

    print("Merging weighted adapter into base weights.")
    merged_model = model.merge_and_unload()
    merged_model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, cache_dir=config.hf_cache_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": config.base_model,
        "lora_adapters": adapters,
        "combination_type": args.combination_type,
        "weights": {"code": args.code_weight, "math": args.math_weight},
        "svd_rank": args.svd_rank,
        "density": args.density,
        "torch_dtype": dtype_name,
        "device_map": device_map,
    }
    with (output_dir / "lora_adapter_merge.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Saved merged LoRA model to {output_dir}")

    del merged_model
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
