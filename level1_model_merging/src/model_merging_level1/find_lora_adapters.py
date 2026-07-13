from __future__ import annotations

import argparse
import json
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def _has_adapter_config(model: Any) -> bool:
    return any(sibling.rfilename == "adapter_config.json" for sibling in (model.siblings or []))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find PEFT/LoRA adapters that declare a specific base model.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B", help="Required base_model_name_or_path.")
    parser.add_argument("--query", action="append", default=[], help="Hugging Face model search query.")
    parser.add_argument("--limit", type=int, default=100, help="Max models to inspect per query.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    queries = args.query or [
        "Qwen2.5-1.5B LoRA math",
        "Qwen2.5-1.5B LoRA code",
        "Qwen/Qwen2.5-1.5B lora",
        "Qwen2.5 1.5B adapter",
    ]

    api = HfApi()
    seen: set[str] = set()
    matches: list[dict[str, Any]] = []
    related_bases: list[tuple[str, str]] = []

    for query in queries:
        print(f"Searching: {query}")
        for model in api.list_models(search=query, limit=args.limit, full=True):
            model_id = model.modelId
            if model_id in seen:
                continue
            seen.add(model_id)
            if not _has_adapter_config(model):
                continue

            try:
                path = hf_hub_download(model_id, "adapter_config.json")
                with open(path, encoding="utf-8") as handle:
                    adapter_config = json.load(handle)
            except Exception as exc:
                print(f"  skip {model_id}: {type(exc).__name__}: {exc}")
                continue

            base = str(adapter_config.get("base_model_name_or_path"))
            if base == args.base_model:
                matches.append(
                    {
                        "model_id": model_id,
                        "base": base,
                        "task_type": adapter_config.get("task_type"),
                        "r": adapter_config.get("r"),
                        "lora_alpha": adapter_config.get("lora_alpha"),
                        "use_dora": adapter_config.get("use_dora"),
                        "target_modules": adapter_config.get("target_modules"),
                    }
                )
            elif "Qwen2.5-1.5B" in base or "Qwen2.5-1.5B" in model_id:
                related_bases.append((model_id, base))

    print("\nMATCHES")
    for match in matches:
        print(json.dumps(match, ensure_ascii=False))
    print(f"match_count={len(matches)} searched={len(seen)}")

    if related_bases:
        print("\nRELATED_NON_MATCHING_BASES")
        for model_id, base in related_bases:
            print(f"{model_id}\t{base}")


if __name__ == "__main__":
    main()
