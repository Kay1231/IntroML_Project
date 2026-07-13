from __future__ import annotations

import argparse
import gc
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


def _materialize_one(
    *,
    alias: str,
    adapter_ref: str,
    base_model: str,
    output_dir: Path,
    hf_cache_dir: str | None,
    dtype_name: str,
    device_map: str,
    force: bool,
) -> None:
    torch, PeftConfig, PeftModel, AutoModelForCausalLM, AutoTokenizer = _load_runtime()

    if output_dir.exists() and not force:
        print(f"Skip {alias}: {output_dir} already exists. Use --force to overwrite.")
        return
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)

    peft_config = PeftConfig.from_pretrained(adapter_ref, cache_dir=hf_cache_dir)
    declared_base = str(peft_config.base_model_name_or_path)
    if declared_base != base_model:
        raise ValueError(
            f"{alias} adapter declares base {declared_base!r}, "
            f"but experiment base is {base_model!r}."
        )

    device_map_value = None if device_map == "none" else device_map
    print(f"Loading base for {alias}: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=_torch_dtype(torch, dtype_name),
        device_map=device_map_value,
        low_cpu_mem_usage=True,
        cache_dir=hf_cache_dir,
    )
    print(f"Loading LoRA adapter for {alias}: {adapter_ref}")
    model = PeftModel.from_pretrained(model, adapter_ref, cache_dir=hf_cache_dir)
    print(f"Merging LoRA into base for {alias}.")
    model = model.merge_and_unload()
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_ref, cache_dir=hf_cache_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=hf_cache_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved {alias} to {output_dir}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize PEFT LoRA experts as full local models.")
    parser.add_argument("--config", default="configs/lora_experiment.yaml", help="Path to LoRA experiment YAML.")
    parser.add_argument("--only", action="append", default=None, help="Materialize only this alias. May be repeated.")
    parser.add_argument("--torch-dtype", default=None, help="auto, bfloat16, float16, or float32.")
    parser.add_argument("--device-map", default=None, help="auto, cuda, cpu, or none.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing local expert directories.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_project_config(args.config)
    raw = _load_raw_config(args.config)
    adapters = dict(raw.get("lora_adapters", {}))
    if not adapters:
        raise ValueError("No lora_adapters section found in config.")

    paths = raw.get("paths", {})
    lora_root = Path(paths.get("lora_expert_dir", "lora_experts"))
    if not lora_root.is_absolute():
        lora_root = config.project_root / lora_root

    selected = set(args.only or adapters.keys())
    unknown = selected.difference(adapters.keys())
    if unknown:
        raise ValueError(f"Unknown LoRA aliases: {sorted(unknown)}")

    gen_config = config.eval.get("generation", {})
    dtype_name = args.torch_dtype or gen_config.get("torch_dtype", "auto")
    device_map = args.device_map or gen_config.get("device_map", "auto")

    for alias in adapters:
        if alias not in selected:
            continue
        output_dir = config.project_root / config.expert_models[alias]
        _materialize_one(
            alias=alias,
            adapter_ref=_resolve_adapter_ref(config.project_root, str(adapters[alias])),
            base_model=config.base_model,
            output_dir=output_dir,
            hf_cache_dir=config.hf_cache_dir,
            dtype_name=dtype_name,
            device_map=device_map,
            force=args.force,
        )


if __name__ == "__main__":
    main()
