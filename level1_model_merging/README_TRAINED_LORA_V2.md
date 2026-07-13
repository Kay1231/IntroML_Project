# 自训练 LoRA v2 实验记录

本轮实验目标是解决 v1 自训练 LoRA 的 prompt / 数据形式不匹配问题：

- math LoRA：训练 prompt 对齐当前 `evaluate.py` 的 GSM8K eval prompt，要求最后一行输出 `#### <number>`。
- code LoRA：改用 MBPP/HumanEval 风格的“给定函数签名，补全函数体”数据，训练输出只包含函数体。
- 训练学习率降到 `5e-5`，减少低资源训练时的格式漂移。

## 关键配置

- 配置文件：`configs/train_lora_v2_experiment.yaml`
- 基座模型：`Qwen/Qwen2.5-1.5B`
- math 数据：`openai/gsm8k`, train, 3000 samples, 2 epochs
- code 数据：`google-research-datasets/mbpp`, full, `train+validation+prompt+test`, 5 epochs
- LoRA rank：16
- 评测规模：GSM8K 100 题，HumanEval 50 题

## 已完成的合并搜索

本轮补齐了三档权重：code/math = 80/20、50/50、20/80。

- MergeKit SLERP：`slerp_v2_code080_math020`, `slerp_v2_code050_math050`, `slerp_v2_code020_math080`
- MergeKit Task Arithmetic：`task_v2_code080_math020_l100`, `task_v2_code050_math050_l100`, `task_v2_code020_math080_l100`
- MergeKit TIES：`ties_v2_code080_math020_d080`, `ties_v2_code050_math050_d080`, `ties_v2_code020_math080_d080`
- PEFT adapter SVD：`lora_svd_v2_code080_math020`, `lora_svd_v2_code050_math050`, `lora_svd_v2_code020_math080`
- PEFT adapter TIES-SVD：`lora_ties_svd_v2_code080_math020`, `lora_ties_svd_v2_code050_math050`, `lora_ties_svd_v2_code020_math080`

## 复现实验命令

从项目根目录运行：

```powershell
cd D:\IntroML\project\level1_model_merging

powershell -ExecutionPolicy Bypass -File .\scripts\run_trained_lora_v2_pipeline.ps1 `
  -LimitMath 100 `
  -LimitCode 50 `
  -SkipExisting
```

如果只想复用已经训练好的 LoRA adapter，不重新训练：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_trained_lora_v2_pipeline.ps1 `
  -LimitMath 100 `
  -LimitCode 50 `
  -SkipTraining `
  -SkipMaterialize `
  -SkipExisting
```

## 当前结果

结果文件：

- `results_trained_lora_v2/results_summary.csv`
- `results_trained_lora_v2/plots/bar_accuracy.png`
- `results_trained_lora_v2/plots/radar_accuracy.png`
- `results_trained_lora_v2/plots/slerp_weight_curve.png`

| model | GSM8K accuracy | HumanEval pass@1 |
|---|---:|---:|
| base | 0.03 | 0.00 |
| expert_code | 0.16 | 0.26 |
| expert_math | 0.61 | 0.34 |
| slerp_v2_code080_math020 | 0.41 | 0.32 |
| slerp_v2_code050_math050 | 0.59 | 0.36 |
| slerp_v2_code020_math080 | 0.58 | 0.38 |
| task_v2_code080_math020_l100 | 0.40 | 0.28 |
| task_v2_code050_math050_l100 | 0.57 | 0.36 |
| task_v2_code020_math080_l100 | 0.60 | 0.38 |
| ties_v2_code080_math020_d080 | 0.51 | 0.18 |
| ties_v2_code050_math050_d080 | 0.54 | 0.28 |
| ties_v2_code020_math080_d080 | 0.56 | 0.36 |
| lora_svd_v2_code080_math020 | 0.48 | 0.04 |
| lora_svd_v2_code050_math050 | 0.53 | 0.20 |
| lora_svd_v2_code020_math080 | 0.52 | 0.36 |
| lora_ties_svd_v2_code080_math020 | 0.42 | 0.14 |
| lora_ties_svd_v2_code050_math050 | 0.56 | 0.28 |
| lora_ties_svd_v2_code020_math080 | 0.61 | 0.36 |

## 报告结论建议

本轮 v2 实验已经满足“融合之后的模型在对应任务上的性能相较于基座模型有显著提升”的 Level 1 目标。

最佳综合合并模型建议写 `task_v2_code020_math080_l100`：

- GSM8K：0.03 -> 0.60
- HumanEval：0.00 -> 0.38
- 这是非 SLERP 的 Task Arithmetic，适合强调“除了 SLERP，Task Arithmetic 在同 base、同 prompt 对齐的 LoRA 专家上也能显著提升”。

最佳 SLERP 模型建议写 `slerp_v2_code020_math080`：

- GSM8K：0.03 -> 0.58
- HumanEval：0.00 -> 0.38
- SLERP 表现稳定，50/50 和 20/80 都强，说明球面插值对 LoRA 物化后的专家权重冲突更稳健。

TIES / TIES-SVD 的结论可以写得更细：

- full-weight TIES 在 20/80 时达到 GSM8K 0.56 / HumanEval 0.36，明显好于 base，但略低于 Task Arithmetic 和 SLERP。
- adapter TIES-SVD 在 20/80 时达到 GSM8K 0.61 / HumanEval 0.36，数学最好，代码也有明显保留。
- code-heavy 权重普遍不是最优，因为本轮 math LoRA 在 HumanEval 上也不弱，20/80 反而同时保留 math 和 code 能力。

可以在报告中强调：v1 失败的主要原因不是合并算法本身，而是专家模型训练分布和评测 prompt 不对齐；v2 通过同一 base 的 LoRA 专家、对齐 eval prompt、函数补全式 code 数据、较低学习率，显著降低了参数冲突，使 SLERP、Task Arithmetic、TIES、LoRA SVD/TIES-SVD 都能在两个任务上相较 base 取得正提升。
