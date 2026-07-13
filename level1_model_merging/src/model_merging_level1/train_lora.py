from __future__ import annotations

import argparse
import gc
import math
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import load_project_config


def _load_runtime():
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    return load_dataset, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def _load_raw_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _torch_dtype(value: str) -> torch.dtype:
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype for training: {value}")


def _task_output_dir(config_path: str | Path, raw: dict[str, Any], task: str) -> Path:
    project_root = Path(config_path).resolve().parent.parent
    adapter_ref = raw.get("lora_adapters", {}).get(task)
    if not adapter_ref:
        raise ValueError(f"No lora_adapters.{task} entry in config.")
    path = Path(str(adapter_ref))
    return path if path.is_absolute() else project_root / path


def _chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"


def _format_math(example: dict[str, Any], tokenizer: Any, format_name: str) -> tuple[str, str]:
    if format_name == "gsm8k_eval_prompt":
        system_prompt = (
            "You are a careful arithmetic word-problem solver. "
            "Do not write Python, pseudocode, or calculator code. "
            "Reason briefly, then put the final numeric answer on the last line as '#### <number>'."
        )
        user_prompt = (
            f"{example['question']}\n\n"
            "Solve the problem. The last line must be exactly: #### <number>"
        )
        return _chat_prompt(tokenizer, system_prompt, user_prompt), str(example["answer"]).strip()

    prompt = (
        "Solve the following grade-school math problem. "
        "Reason step by step and end with a line in the form #### <number>.\n\n"
        f"Problem:\n{example['question']}\n\nSolution:\n"
    )
    return prompt, str(example["answer"]).strip()


def _strip_leading_docstring(body_lines: list[str], function_node: Any) -> list[str]:
    if not function_node.body:
        return body_lines
    first = function_node.body[0]
    if not (
        first.__class__.__name__ == "Expr"
        and getattr(getattr(first, "value", None), "value", None).__class__ is str
    ):
        return body_lines

    docstring_end = int(getattr(first, "end_lineno", first.lineno))
    body_start = int(function_node.body[0].lineno)
    drop_count = max(0, docstring_end - body_start + 1)
    return body_lines[drop_count:]


def _pick_mbpp_function(tree: Any, tests: str) -> Any:
    functions = [node for node in tree.body if node.__class__.__name__ == "FunctionDef"]
    if not functions:
        raise ValueError("MBPP code example does not contain a top-level function.")

    def score(node: Any) -> int:
        return len(re.findall(rf"\b{re.escape(node.name)}\s*\(", tests))

    return max(functions, key=score)


def _format_mbpp_function_body(example: dict[str, Any], tokenizer: Any) -> tuple[str, str]:
    import ast

    raw_code = str(example["code"]).replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    code = textwrap.dedent(raw_code).strip("\n")
    tree = ast.parse(code)
    tests = "\n".join(str(item) for item in example.get("test_list", []) or [])
    function_node = _pick_mbpp_function(tree, tests)
    lines = code.splitlines()

    start = int(function_node.lineno) - 1
    end = int(getattr(function_node, "end_lineno", function_node.lineno))
    signature = lines[start].strip()
    prelude = "\n".join(lines[:start]).strip()

    body_start = int(function_node.body[0].lineno) - 1 if function_node.body else start + 1
    body_lines = lines[body_start:end]
    body_lines = _strip_leading_docstring(body_lines, function_node)
    response = "\n".join(body_lines).rstrip() or "    pass"

    problem = str(example.get("prompt") or example.get("text") or "").strip()
    docstring = '    """' + problem.replace('"""', "'''") + '"""'
    function_prompt_parts = []
    if prelude:
        function_prompt_parts.append(prelude)
    function_prompt_parts.append(signature)
    function_prompt_parts.append(docstring)
    function_prompt = "\n".join(function_prompt_parts)

    system_prompt = (
        "You are an expert Python programmer. "
        "Return only valid Python code for the requested function. "
        "Do not include explanations, tests, markdown fences, or print statements."
    )
    user_prompt = (
        "Complete the following Python function. "
        "Return only the function body, or the full function definition if needed.\n\n"
        f"{function_prompt}"
    )
    return _chat_prompt(tokenizer, system_prompt, user_prompt), response


def _format_code(example: dict[str, Any], tokenizer: Any, format_name: str) -> tuple[str, str]:
    if format_name == "mbpp_function_body":
        return _format_mbpp_function_body(example, tokenizer)

    instruction = str(example.get("instruction") or example.get("prompt") or "").strip()
    problem_input = str(example.get("input") or "").strip()
    response = str(example.get("output") or example.get("completion") or "").strip()
    if problem_input:
        prompt = (
            "Write a correct Python solution for the following programming task.\n\n"
            f"Task:\n{instruction}\n\nInput:\n{problem_input}\n\nPython solution:\n"
        )
    else:
        prompt = (
            "Write a correct Python solution for the following programming task.\n\n"
            f"Task:\n{instruction}\n\nPython solution:\n"
        )
    return prompt, response


def _format_example(
    task: str,
    example: dict[str, Any],
    tokenizer: Any,
    task_config: dict[str, Any],
) -> tuple[str, str]:
    format_name = str(task_config.get("format", "default"))
    if task == "math":
        return _format_math(example, tokenizer, format_name)
    if task == "code":
        return _format_code(example, tokenizer, format_name)
    raise ValueError(f"Unsupported task: {task}")


def _tokenize_example(
    *,
    tokenizer: Any,
    task: str,
    example: dict[str, Any],
    task_config: dict[str, Any],
    max_seq_length: int,
) -> dict[str, Any]:
    prompt, response = _format_example(task, example, tokenizer, task_config)
    full_text = prompt + response + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized["input_ids"]
    labels = list(input_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    if all(label == -100 for label in labels) and labels:
        labels[-1] = input_ids[-1]

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _select_dataset(raw_dataset: Any, max_samples: int, seed: int) -> Any:
    if max_samples <= 0 or max_samples >= len(raw_dataset):
        return raw_dataset
    indices = list(range(len(raw_dataset)))
    random.Random(seed).shuffle(indices)
    return raw_dataset.select(indices[:max_samples])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a task LoRA for the Level 1 model merging project.")
    parser.add_argument("--config", default="configs/train_lora_experiment.yaml", help="Path to training config.")
    parser.add_argument("--task", choices=["math", "code"], required=True, help="Which LoRA to train.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override configured sample count.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override configured max_steps. 0 means epoch-based.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override configured max sequence length.")
    parser.add_argument("--output-dir", default=None, help="Override adapter output directory.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint path or 'true'.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny 2-step training test.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_project_config(args.config)
    raw = _load_raw_config(args.config)
    train_config = raw.get("train", {})
    common = dict(train_config.get("common", {}))
    task_config = dict(train_config.get("tasks", {}).get(args.task, {}))
    if not task_config:
        raise ValueError(f"No train.tasks.{args.task} section in config.")
    train_settings = dict(common)
    train_settings.update(task_config)

    load_dataset, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = _load_runtime()

    seed = int(common.get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)

    output_dir = Path(args.output_dir) if args.output_dir else _task_output_dir(args.config, raw, args.task)
    if args.smoke_test:
        output_dir = config.project_root / "runs" / f"smoke_{args.task}_lora"

    max_samples = args.max_samples if args.max_samples is not None else int(task_config.get("max_samples", 0))
    max_steps = args.max_steps if args.max_steps is not None else int(train_settings.get("max_steps", 0))
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else int(train_settings.get("max_seq_length", 512))
    if args.smoke_test:
        max_samples = min(max_samples or 16, 16)
        max_steps = 2
        max_seq_length = min(max_seq_length, 384)

    print(f"Loading tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, cache_dir=config.hf_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset for {args.task}: {task_config['dataset_name']}")
    dataset_kwargs: dict[str, Any] = {}
    if task_config.get("dataset_config"):
        dataset_kwargs["name"] = task_config["dataset_config"]
    raw_dataset = load_dataset(
        task_config["dataset_name"],
        **dataset_kwargs,
        split=task_config.get("split", "train"),
        cache_dir=config.hf_cache_dir,
    )
    raw_dataset = _select_dataset(raw_dataset, max_samples=max_samples, seed=seed)
    print(f"Training examples: {len(raw_dataset)}")

    tokenized_dataset = raw_dataset.map(
        lambda example: _tokenize_example(
            tokenizer=tokenizer,
            task=args.task,
            example=example,
            task_config=task_config,
            max_seq_length=max_seq_length,
        ),
        remove_columns=raw_dataset.column_names,
        desc=f"Tokenizing {args.task}",
    )

    dtype = _torch_dtype(str(train_settings.get("torch_dtype", "bfloat16")))
    print(f"Loading base model: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=config.hf_cache_dir,
    )
    model.config.use_cache = False
    if bool(common.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(train_settings.get("lora_r", 16)),
        lora_alpha=int(train_settings.get("lora_alpha", 32)),
        lora_dropout=float(train_settings.get("lora_dropout", 0.05)),
        target_modules=list(train_settings.get("target_modules", [])),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    steps_per_epoch = max(1, math.ceil(len(tokenized_dataset) / int(train_settings.get("gradient_accumulation_steps", 8))))
    print(f"Approx optimizer steps per epoch: {steps_per_epoch}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_settings.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_settings.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_settings.get("learning_rate", 2e-4)),
        num_train_epochs=float(train_settings.get("num_train_epochs", 1)),
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        warmup_ratio=float(train_settings.get("warmup_ratio", 0.03)),
        logging_steps=int(train_settings.get("logging_steps", 10)),
        save_steps=int(train_settings.get("save_steps", 200)),
        save_total_limit=int(train_settings.get("save_total_limit", 2)),
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        do_train=True,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )

    resume_value: str | bool | None = args.resume_from_checkpoint
    if resume_value == "true":
        resume_value = True

    print(f"Training {args.task} LoRA -> {output_dir}")
    trainer.train(resume_from_checkpoint=resume_value)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print(f"Saved {args.task} LoRA adapter to {output_dir}")

    del trainer
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
