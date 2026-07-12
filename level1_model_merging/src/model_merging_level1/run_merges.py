from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from .config import load_project_config, select_methods
from .merge_configs import write_merge_configs


def _mergekit_command(
    merge_config: Path,
    output_dir: Path,
    use_cuda: bool,
    lazy_unpickle: bool,
    allow_crimes: bool,
    transformers_cache: str | None,
) -> list[str]:
    command = ["mergekit-yaml", str(merge_config), str(output_dir)]
    if use_cuda:
        command.append("--cuda")
    if lazy_unpickle:
        command.append("--lazy-unpickle")
    if allow_crimes:
        command.append("--allow-crimes")
    if transformers_cache:
        command.extend(["--transformers-cache", transformers_cache])
    return command


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MergeKit merges for the Level 1 experiment.")
    parser.add_argument("--config", default="configs/default_experiment.yaml", help="Path to the experiment YAML.")
    parser.add_argument("--only", action="append", default=None, help="Run only this method. May be repeated.")
    parser.add_argument("--cuda", action="store_true", help="Pass --cuda to mergekit-yaml.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing merged model directories.")
    parser.add_argument("--no-lazy-unpickle", action="store_true", help="Do not pass --lazy-unpickle to mergekit-yaml.")
    parser.add_argument("--allow-crimes", action="store_true", help="Pass --allow-crimes for unusual tokenizer/model cases.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--retries", type=int, default=3, help="Retry a failed merge command this many times.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_project_config(args.config)
    write_merge_configs(config, args.only)

    for method in select_methods(config, args.only):
        name = str(method["name"])
        merge_config = config.merge_config_dir / f"{name}.yaml"
        output_dir = config.merged_model_dir(name)

        if output_dir.exists():
            if not args.force:
                print(f"Skip {name}: {output_dir} already exists. Use --force to overwrite.")
                continue
            shutil.rmtree(output_dir)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        command = _mergekit_command(
            merge_config=merge_config,
            output_dir=output_dir,
            use_cuda=args.cuda,
            lazy_unpickle=not args.no_lazy_unpickle,
            allow_crimes=args.allow_crimes,
            transformers_cache=config.hf_cache_dir,
        )
        print("Running:", " ".join(command))
        if not args.dry_run:
            for attempt in range(1, args.retries + 1):
                try:
                    subprocess.run(command, check=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt >= args.retries:
                        raise
                    wait_seconds = min(60, 10 * attempt)
                    print(f"{name} failed on attempt {attempt}/{args.retries}; retrying in {wait_seconds}s.")
                    time.sleep(wait_seconds)


if __name__ == "__main__":
    main()
