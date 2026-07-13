from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .config import ProjectConfig, load_project_config


NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+|\d+/\d+)")
FENCED_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
GENERIC_FENCED_BLOCK_PATTERN = re.compile(r"```\s*\n?(?P<code>.*?)```", flags=re.DOTALL)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_ref: str


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _summary_path(config: ProjectConfig, model_alias: str, benchmark: str) -> Path:
    return config.eval_dir() / f"{_safe_name(model_alias)}_{benchmark}_summary.json"


def _load_runtime():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, load_dataset, AutoModelForCausalLM, AutoTokenizer


def _torch_dtype(torch_module: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    if not hasattr(torch_module, value):
        raise ValueError(f"Unknown torch dtype: {value}")
    return getattr(torch_module, value)


def _chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _fraction_from_text(value: str) -> Fraction | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        if "/" in cleaned and cleaned.count("/") == 1:
            return Fraction(cleaned)
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def _last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text.replace("$", ""))
    return matches[-1] if matches else None


def _strip_fenced_blocks(text: str) -> str:
    text = FENCED_BLOCK_PATTERN.sub("\n", text)
    return GENERIC_FENCED_BLOCK_PATTERN.sub("\n", text)


def extract_gsm8k_answer(answer_text: str) -> str | None:
    marker = "####"
    if marker in answer_text:
        tail = answer_text.split(marker)[-1]
        return _last_number(tail)
    return _last_number(answer_text)


def extract_generated_number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        number = _last_number(boxed[-1])
        if number is not None:
            return number

    final_patterns = [
        r"####\s*([^\n]+)",
        r"final\s+answer\s*(?:is|=|:)?\s*([^\n]+)",
        r"answer\s*(?:is|=|:)?\s*([^\n]+)",
        r"therefore\s*,?\s*(?:the\s+answer\s+is\s*)?([^\n]+)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            number = _last_number(matches[-1])
            if number is not None:
                return number

    text_without_code = _strip_fenced_blocks(text)
    return _last_number(text_without_code)


def numeric_equal(predicted: str | None, target: str | None) -> bool:
    if predicted is None or target is None:
        return False
    left = _fraction_from_text(predicted)
    right = _fraction_from_text(target)
    if left is None or right is None:
        return False
    if left == right:
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
    except OverflowError:
        return False


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    load_dataset: Any,
    config: ProjectConfig,
    model_alias: str,
    limit: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    math_config = config.eval.get("math", {})
    dataset = load_dataset(
        math_config.get("dataset_name", "openai/gsm8k"),
        math_config.get("dataset_config", "main"),
        split=math_config.get("split", "test"),
        cache_dir=config.hf_cache_dir,
    )
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    details_path = config.eval_dir() / f"{_safe_name(model_alias)}_gsm8k.jsonl"
    rows = _load_jsonl_rows(details_path)
    start_index = min(len(rows), len(dataset))
    if start_index:
        print(f"Resume {model_alias} GSM8K from {start_index}/{len(dataset)} existing rows.")

    correct = sum(int(bool(row.get("correct"))) for row in rows[:start_index])
    system_prompt = (
        "You are a careful arithmetic word-problem solver. "
        "Do not write Python, pseudocode, or calculator code. "
        "Reason briefly, then put the final numeric answer on the last line as '#### <number>'."
    )

    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("a" if start_index else "w", encoding="utf-8") as handle:
        for index, item in enumerate(tqdm(dataset, desc=f"{model_alias} GSM8K")):
            if index < start_index:
                continue

            question = item["question"]
            prompt = _chat_prompt(
                tokenizer,
                system_prompt,
                f"{question}\n\nSolve the problem. The last line must be exactly: #### <number>",
            )
            generation = _generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            prediction = extract_generated_number(generation)
            target = extract_gsm8k_answer(item["answer"])
            is_correct = numeric_equal(prediction, target)
            correct += int(is_correct)
            row = {
                "question": question,
                "target": target,
                "prediction": prediction,
                "correct": is_correct,
                "generation": generation,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    total = len(rows)
    return {
        "model": model_alias,
        "benchmark": "gsm8k",
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "details_path": str(details_path),
    }


def _extract_python_code(text: str) -> str:
    completion = text.strip()
    if not completion:
        return ""

    matches = [match.group("code").strip("\n") for match in FENCED_BLOCK_PATTERN.finditer(completion)]
    if not matches:
        matches = [match.group("code").strip("\n") for match in GENERIC_FENCED_BLOCK_PATTERN.finditer(completion)]
    if matches:
        return max(matches, key=len).strip()
    return completion


def _trim_code_after_solution(code: str) -> str:
    stop_patterns = [
        r"(?m)^\s*if\s+__name__\s*==",
        r"(?m)^\s*print\s*\(",
        r"(?m)^\s*assert\s+",
        r"(?m)^def\s+check\s*\(",
        r"(?m)^check\s*\(",
        r"(?m)^#\s*test",
        r"(?m)^#\s*example",
    ]
    trimmed = code
    for pattern in stop_patterns:
        match = re.search(pattern, trimmed, flags=re.IGNORECASE)
        if match:
            trimmed = trimmed[: match.start()]
    return trimmed.strip("\n")


def _contains_entry_point_definition(code: str, entry_point: str) -> bool:
    pattern = rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\("
    return re.search(pattern, code) is not None


def _ensure_function_body_indented(code: str) -> str:
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    if not lines:
        return "    pass\n"

    first_nonempty = next((line for line in lines if line.strip()), "")
    if first_nonempty.startswith((" ", "\t")):
        return "\n".join(lines).rstrip() + "\n"

    return "\n".join(("    " + line if line.strip() else line) for line in lines).rstrip() + "\n"


def clean_code_completion(text: str, entry_point: str | None = None) -> str:
    code = _extract_python_code(text)
    code = _trim_code_after_solution(code)
    if entry_point and _contains_entry_point_definition(code, entry_point):
        return code.rstrip() + "\n"
    return _ensure_function_body_indented(code)


def _build_humaneval_program(prompt: str, completion: str, tests: str, entry_point: str) -> str:
    code = _extract_python_code(completion)
    code = _trim_code_after_solution(code)
    if _contains_entry_point_definition(code, entry_point):
        candidate = code.rstrip() + "\n"
    else:
        candidate = prompt.rstrip() + "\n" + _ensure_function_body_indented(code)
    return f"{candidate}\n{tests}\n\ncheck({entry_point})\n"


def _syntax_ok(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def _run_python_program(source: str, timeout_seconds: int = 8) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="humaneval_") as temp_dir:
        program_path = Path(temp_dir) / "candidate.py"
        program_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(program_path)],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    passed = result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    return passed, output[-4000:]


def evaluate_humaneval(
    model: Any,
    tokenizer: Any,
    load_dataset: Any,
    config: ProjectConfig,
    model_alias: str,
    limit: int,
    max_new_tokens: int,
    allow_code_execution: bool,
) -> dict[str, Any]:
    code_config = config.eval.get("code", {})
    dataset = load_dataset(
        code_config.get("dataset_name", "openai/openai_humaneval"),
        split=code_config.get("split", "test"),
        cache_dir=config.hf_cache_dir,
    )
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    details_path = config.eval_dir() / f"{_safe_name(model_alias)}_humaneval.jsonl"
    rows = _load_jsonl_rows(details_path)
    start_index = min(len(rows), len(dataset))
    if start_index:
        print(f"Resume {model_alias} HumanEval from {start_index}/{len(dataset)} existing rows.")

    passed_count = sum(int(bool(row.get("passed"))) for row in rows[:start_index])
    syntax_count = sum(int(bool(row.get("syntax_ok"))) for row in rows[:start_index])
    system_prompt = (
        "You are an expert Python programmer. "
        "Return only valid Python code for the requested function. "
        "Do not include explanations, tests, markdown fences, or print statements."
    )

    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("a" if start_index else "w", encoding="utf-8") as handle:
        for index, item in enumerate(tqdm(dataset, desc=f"{model_alias} HumanEval")):
            if index < start_index:
                continue

            prompt_text = item["prompt"]
            prompt = _chat_prompt(
                tokenizer,
                system_prompt,
                (
                    "Complete the following Python function. "
                    "Return only the function body, or the full function definition if needed.\n\n"
                    f"{prompt_text}"
                ),
            )
            generation = _generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            program = _build_humaneval_program(prompt_text, generation, item["test"], item["entry_point"])
            syntax_ok = _syntax_ok(program)
            syntax_count += int(syntax_ok)
            passed = None
            execution_output = ""

            if allow_code_execution:
                try:
                    passed, execution_output = _run_python_program(program)
                except subprocess.TimeoutExpired:
                    passed, execution_output = False, "Timed out"
                passed_count += int(bool(passed))

            row = {
                "task_id": item["task_id"],
                "entry_point": item["entry_point"],
                "syntax_ok": syntax_ok,
                "passed": passed,
                "generation": generation,
                "program": program,
                "execution_output": execution_output,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    total = len(rows)
    summary = {
        "model": model_alias,
        "benchmark": "humaneval",
        "num_examples": total,
        "syntax_rate": syntax_count / total if total else 0.0,
        "syntax_ok": syntax_count,
        "details_path": str(details_path),
        "allow_code_execution": allow_code_execution,
    }
    if allow_code_execution:
        summary["pass_at_1"] = passed_count / total if total else 0.0
        summary["passed"] = passed_count
    return summary


def collect_model_specs(config: ProjectConfig, args: argparse.Namespace) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    if args.include_base:
        specs.append(ModelSpec("base", config.base_model))
    if args.include_experts:
        specs.append(ModelSpec("expert_code", config.expert_models["code"]))
        specs.append(ModelSpec("expert_math", config.expert_models["math"]))
    if args.include_merges:
        for method in [*config.methods, *config.adapter_merges]:
            name = str(method["name"])
            path = config.merged_model_dir(name)
            if path.exists():
                specs.append(ModelSpec(name, str(path)))
            else:
                print(f"Skip merged model {name}: {path} does not exist.")
    for item in args.model:
        if "=" not in item:
            raise ValueError("--model must use alias=path_or_hf_id format")
        alias, ref = item.split("=", 1)
        specs.append(ModelSpec(alias, ref))

    if not specs:
        raise ValueError("No models selected. Use --include-base, --include-experts, --include-merges, or --model alias=ref.")
    return specs


def evaluate_model(
    spec: ModelSpec,
    config: ProjectConfig,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    torch, load_dataset, AutoModelForCausalLM, AutoTokenizer = _load_runtime()
    gen_config = config.eval.get("generation", {})
    dtype_name = args.torch_dtype or gen_config.get("torch_dtype", "auto")
    device_map = args.device_map or gen_config.get("device_map", "auto")
    device_map_value = None if device_map == "none" else device_map

    tokenizer = AutoTokenizer.from_pretrained(spec.model_ref, cache_dir=config.hf_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.model_ref,
        torch_dtype=_torch_dtype(torch, dtype_name),
        device_map=device_map_value,
        low_cpu_mem_usage=True,
        cache_dir=config.hf_cache_dir,
    )
    model.eval()

    summaries: list[dict[str, Any]] = []
    if "gsm8k" in args.benchmark:
        summaries.append(
            evaluate_gsm8k(
                model=model,
                tokenizer=tokenizer,
                load_dataset=load_dataset,
                config=config,
                model_alias=spec.alias,
                limit=args.limit_math,
                max_new_tokens=args.max_new_tokens_math or gen_config.get("max_new_tokens_math", 768),
            )
        )
    if "humaneval" in args.benchmark:
        summaries.append(
            evaluate_humaneval(
                model=model,
                tokenizer=tokenizer,
                load_dataset=load_dataset,
                config=config,
                model_alias=spec.alias,
                limit=args.limit_code,
                max_new_tokens=args.max_new_tokens_code or gen_config.get("max_new_tokens_code", 512),
                allow_code_execution=args.allow_code_execution,
            )
        )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate base, expert, and merged models.")
    parser.add_argument("--config", default="configs/default_experiment.yaml", help="Path to the experiment YAML.")
    parser.add_argument("--include-base", action="store_true", help="Evaluate the base model.")
    parser.add_argument("--include-experts", action="store_true", help="Evaluate the source expert models.")
    parser.add_argument("--include-merges", action="store_true", help="Evaluate existing merged model directories.")
    parser.add_argument("--model", action="append", default=[], help="Extra model in alias=path_or_hf_id format.")
    parser.add_argument("--benchmark", action="append", choices=["gsm8k", "humaneval"], default=[])
    parser.add_argument("--limit-math", type=int, default=None, help="Number of GSM8K examples. 0 means all.")
    parser.add_argument("--limit-code", type=int, default=None, help="Number of HumanEval examples. 0 means all.")
    parser.add_argument("--max-new-tokens-math", type=int, default=None)
    parser.add_argument("--max-new-tokens-code", type=int, default=None)
    parser.add_argument("--torch-dtype", default=None, help="auto, float16, bfloat16, or float32.")
    parser.add_argument("--device-map", default=None, help="auto, cuda, cpu, or none.")
    parser.add_argument("--allow-code-execution", action="store_true", help="Run generated HumanEval code to compute pass@1.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not recompute existing summary JSON files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.benchmark:
        args.benchmark = ["gsm8k", "humaneval"]

    config = load_project_config(args.config)
    math_limit = config.eval.get("math", {}).get("limit", 100)
    code_limit = config.eval.get("code", {}).get("limit", 50)
    args.limit_math = math_limit if args.limit_math is None else args.limit_math
    args.limit_code = code_limit if args.limit_code is None else args.limit_code

    config.eval_dir().mkdir(parents=True, exist_ok=True)
    specs = collect_model_specs(config, args)
    for spec in specs:
        original_benchmarks = list(args.benchmark)
        pending_benchmarks = [
            benchmark
            for benchmark in original_benchmarks
            if not args.skip_existing or not _summary_path(config, spec.alias, benchmark).exists()
        ]
        if not pending_benchmarks:
            print(f"Skip {spec.alias}: all requested summary files already exist.")
            continue

        args.benchmark = pending_benchmarks
        print(f"Evaluating {spec.alias}: {spec.model_ref}")
        summaries = evaluate_model(spec, config, args)
        args.benchmark = original_benchmarks

        for summary in summaries:
            path = _summary_path(config, spec.alias, summary["benchmark"])
            path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
