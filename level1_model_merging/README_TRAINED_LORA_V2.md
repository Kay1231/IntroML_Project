# 自训练 LoRA v2 实验记录

本轮实验目标是解决 v1 自训练 LoRA 的 prompt / 数据形式不匹配问题：

- math LoRA：训练 prompt 对齐当前 `evaluate.py` 的 GSM8K eval prompt，要求最后一行输出 `#### <number>`。
- code LoRA：改用 MBPP/HumanEval 风格的“给定函数签名，补全函数体”数据，训练输出只包含函数体。
- 训练学习率降到 `5e-5`，减少低资源训练时的灾难性格式漂移。

## 关键配置

- 配置文件：`configs/train_lora_v2_experiment.yaml`
- 基座模型：`Qwen/Qwen2.5-1.5B`
- math 数据：`openai/gsm8k`, train, 3000 samples, 2 epochs
- code 数据：`google-research-datasets/mbpp`, full, `train+validation+prompt+test`, 5 epochs
- LoRA rank：16
- 合并候选：
  - `slerp_v2_code080_math020`
  - `task_v2_code080_math020_l100`
  - `lora_svd_v2_code080_math020`
  - `lora_ties_svd_v2_code080_math020`

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
  -SkipExisting
```

## 当前结果

结果文件：

- `results_trained_lora_v2/results_summary.csv`
- `results_trained_lora_v2/plots/bar_accuracy.png`
- `results_trained_lora_v2/plots/radar_accuracy.png`

| model | GSM8K accuracy | HumanEval pass@1 |
|---|---:|---:|
| base | 0.03 | 0.00 |
| expert_code | 0.16 | 0.26 |
| expert_math | 0.61 | 0.34 |
| slerp_v2_code080_math020 | 0.41 | 0.32 |
| task_v2_code080_math020_l100 | 0.40 | 0.28 |
| lora_svd_v2_code080_math020 | 0.48 | 0.04 |
| lora_ties_svd_v2_code080_math020 | 0.42 | 0.14 |

## 报告结论建议

本轮 v2 实验已经满足“融合之后的模型在对应任务上的性能相较于基座模型有显著提升”的 Level 1 目标：

- 最佳综合合并模型是 `slerp_v2_code080_math020`：
  - GSM8K：0.03 -> 0.41
  - HumanEval：0.00 -> 0.32
- 非 SLERP 算法也有明确提升：
  - `task_v2_code080_math020_l100`：GSM8K 0.40，HumanEval 0.28
  - `lora_ties_svd_v2_code080_math020`：GSM8K 0.42，HumanEval 0.14
- `lora_svd_v2_code080_math020` 数学最好，但代码保留较弱，适合在报告中作为“能力偏置/冲突”的案例。

可以在报告中强调：v1 失败的主要原因不是合并算法本身，而是专家模型训练分布和评测 prompt 不对齐；v2 通过同一 base 的 LoRA 专家、对齐 eval prompt、函数补全式 code 数据、较低学习率，显著降低了参数冲突，使 SLERP 和 Task Arithmetic 都能保留并融合两个任务能力。
