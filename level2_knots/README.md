# Level 2: KnOTS Reproduction

This folder contains a lightweight reproduction of **Model merging with SVD to tie the knots** (ICLR 2025).

- Paper: <https://arxiv.org/abs/2410.19735>
- Official code: <https://github.com/gstoica27/KnOTS>

## Scope

The original paper evaluates KnOTS on larger vision and language model settings. For this course project and an 8GB RTX 4060 Laptop GPU, this reproduction keeps the core algorithm but uses the already trained Level 1 Qwen2.5-1.5B LoRA v2 experts:

- base: `Qwen/Qwen2.5-1.5B`
- code LoRA: `level1_model_merging/trained_lora_adapters_v2/code_qwen25_1p5b_mbpp_body_lr5e5`
- math LoRA: `level1_model_merging/trained_lora_adapters_v2/math_qwen25_1p5b_gsm8k_evalprompt_lr5e5`

## Implemented Core Logic

For every shared LoRA target weight:

1. Recover each task update as `Delta_i = B_i @ A_i * alpha_i / r_i`.
2. Build the concatenated update matrix `[Delta_code, Delta_math]`.
3. Compute a compact SVD from the low-rank LoRA factors.
4. Project each task update into the shared KnOTS SVD coordinate system.
5. Merge the aligned task coordinates with either task arithmetic or TIES.
6. Reconstruct the merged delta and write it directly into the base model.

This matches the paper's central idea: perform merging after SVD-based alignment so that task vectors are compared in a less entangled coordinate system.

## Run

From `D:\IntroML\project\level2_knots`:

```powershell
.\scripts\run_knots_pipeline.ps1 -LimitMath 100 -LimitCode 50
```

This default run evaluates only the Level 2 KnOTS merged models. Add `-IncludeBaseExperts` if you also want to re-evaluate the base and materialized Level 1 LoRA experts.

For a quick smoke run:

```powershell
.\scripts\run_knots_pipeline.ps1 -Only knots_ties_code020_math080_d080 -LimitMath 5 -LimitCode 2
```

Outputs are written only under this Level 2 folder:

- merged models: `merged_models/`
- evaluation JSONL and summaries: `results/eval/`
- summary CSV: `results/results_summary.csv`
- plots: `results/plots/`
