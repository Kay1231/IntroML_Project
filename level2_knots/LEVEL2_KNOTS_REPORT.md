# Level 2 复现记录：Model merging with SVD to tie the KnOTS

## 1. 论文与复现选择

本项目 Level 2 选择复现论文 **Model merging with SVD to tie the KnOTS**（ICLR 2025）。

- 论文：<https://arxiv.org/abs/2410.19735>
- 官方代码：<https://github.com/gstoica27/KnOTS>

课程要求允许“跑通开源代码，或根据论文公式自己实现核心合并逻辑，并在至少一个数据集上复现论文报告的性能”。由于官方实验主要面向更大的 ViT/Llama3-8B 等设置，直接完整复现论文原始大表格不适合当前 RTX 4060 Laptop 8GB 显存环境。因此本项目采用第二种路线：根据论文核心公式实现 KnOTS 的 SVD 对齐合并逻辑，并在已经稳定的 Qwen2.5-1.5B LoRA 专家上验证。

## 2. 实验设置

复现目录：

```text
D:\IntroML\project\level2_knots
```

复用 Level 1 中已经训练好的 LoRA v2 专家：

| 角色 | 模型/路径 |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B` |
| Code LoRA | `D:\IntroML\project\level1_model_merging\trained_lora_adapters_v2\code_qwen25_1p5b_mbpp_body_lr5e5` |
| Math LoRA | `D:\IntroML\project\level1_model_merging\trained_lora_adapters_v2\math_qwen25_1p5b_gsm8k_evalprompt_lr5e5` |

评测数据集：

| Benchmark | 样本数 | 指标 |
|---|---:|---|
| GSM8K | 100 | accuracy |
| HumanEval | 50 | pass@1，启用代码执行 |

## 3. 核心实现

实现文件：

```text
D:\IntroML\project\level2_knots\src\level2_knots\knots_merge_lora.py
```

对每个共享 LoRA 目标权重，本实现执行以下步骤：

1. 从 LoRA 权重恢复每个任务的 full update：

   ```text
   Delta_i = B_i @ A_i * alpha_i / r_i
   ```

2. 构造拼接矩阵：

   ```text
   [Delta_code, Delta_math]
   ```

3. 使用 LoRA 低秩因子做紧凑 SVD，避免显式对巨大的 full matrix 做昂贵分解。
4. 将 code/math task update 投影到共享 SVD 坐标系。
5. 在对齐后的坐标中执行 Task Arithmetic 或 TIES。
6. 将合并后的 delta 写回 base model 权重并保存完整模型。

这对应论文的核心思想：先通过 SVD/正交基对齐 task update，再进行参数合并，以减少不同任务参数空间错位带来的干扰。

## 4. 已运行命令

合并 KnOTS-TIES：

```powershell
cd D:\IntroML\project\level2_knots
$env:PYTHONPATH='D:\IntroML\project\level2_knots\src;D:\IntroML\project\level1_model_merging\src'
$env:HF_HOME='D:\IntroML\project\hf_home'
$env:HF_DATASETS_CACHE='D:\IntroML\project\hf_datasets_cache'
$env:HF_HUB_CACHE='D:\IntroML\project\hf_cache'
conda run -n IntroML python -u -m level2_knots.knots_merge_lora --config D:\IntroML\project\level2_knots\configs\knots_qwen_lora_v2.yaml --name knots_ties_code020_math080_d080 --method ties --code-weight 0.20 --math-weight 0.80 --density 0.80 --torch-dtype bfloat16 --force
```

合并 KnOTS-TaskArithmetic：

```powershell
conda run -n IntroML python -u -m level2_knots.knots_merge_lora --config D:\IntroML\project\level2_knots\configs\knots_qwen_lora_v2.yaml --name knots_ta_code020_math080 --method task_arithmetic --code-weight 0.20 --math-weight 0.80 --density 1.00 --torch-dtype bfloat16 --force
```

评测：

```powershell
conda run -n IntroML python -m model_merging_level1.evaluate --config D:\IntroML\project\level2_knots\configs\knots_qwen_lora_v2.yaml --model knots_ties_code020_math080_d080=D:\IntroML\project\level2_knots\merged_models\knots_ties_code020_math080_d080 --benchmark gsm8k --benchmark humaneval --limit-math 100 --limit-code 50 --allow-code-execution --skip-existing

conda run -n IntroML python -m model_merging_level1.evaluate --config D:\IntroML\project\level2_knots\configs\knots_qwen_lora_v2.yaml --model knots_ta_code020_math080=D:\IntroML\project\level2_knots\merged_models\knots_ta_code020_math080 --benchmark gsm8k --benchmark humaneval --limit-math 100 --limit-code 50 --allow-code-execution --skip-existing
```

汇总：

```powershell
conda run -n IntroML python -m model_merging_level1.summarize --config D:\IntroML\project\level2_knots\configs\knots_qwen_lora_v2.yaml
```

## 5. 当前结果

结果文件：

```text
D:\IntroML\project\level2_knots\results\results_summary.csv
```

| Model | GSM8K accuracy | HumanEval pass@1 |
|---|---:|---:|
| knots_ta_code020_math080 | 0.62 | 0.40 |
| knots_ties_code020_math080_d080 | 0.61 | 0.44 |

图表：

```text
D:\IntroML\project\level2_knots\results\plots\bar_accuracy.png
D:\IntroML\project\level2_knots\results\plots\radar_accuracy.png
```

## 6. 与 Level 1 最强结果的对比

Level 1 LoRA v2 关键基线：

| Model | GSM8K accuracy | HumanEval pass@1 |
|---|---:|---:|
| base | 0.03 | 0.00 |
| lora_ties_svd_v2_code020_math080 | 0.61 | 0.36 |
| task_v2_code020_math080_l100 | 0.60 | 0.38 |
| slerp_v2_code020_math080 | 0.58 | 0.38 |

观察：

- KnOTS-TA 达到 GSM8K 0.62，比 Level 1 的 full-weight Task Arithmetic 0.60 略高，同时 HumanEval 从 0.38 提高到 0.40。
- KnOTS-TIES 达到 GSM8K 0.61，保持了 Level 1 最强 LoRA TIES-SVD 的数学能力，同时 HumanEval 从 0.36 提高到 0.44。
- 相比 base，两个 KnOTS 合并模型都在两个任务上显著提升：GSM8K 从 0.03 提升到 0.61/0.62，HumanEval 从 0.00 提升到 0.40/0.44。

## 7. 初步结论

本次 Level 2 复现已经跑通了一个基于论文公式的 KnOTS 核心实现，并在 GSM8K 与 HumanEval 两个数据集上得到有效结果。实验支持论文的核心观点：在合并前对 task update 做 SVD 坐标对齐，可以缓解任务间参数干扰。尤其在 HumanEval 上，KnOTS-TIES 相比 Level 1 的 LoRA TIES-SVD 有更明显提升，说明对齐后的 TIES 比直接在原始 LoRA 空间进行 TIES-SVD 更适合当前 code/math LoRA 专家组合。

