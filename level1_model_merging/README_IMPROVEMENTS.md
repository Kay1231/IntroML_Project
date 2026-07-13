# Level 1 改进与最终快速流程

本项目实现课程 Level 1：使用 MergeKit 合并 Code 专家与 Math 专家，并在 GSM8K 与 HumanEval 上验证合并模型是否相对 base 保留/复用了两侧能力。

## 当前推荐配置

主配置为：

```text
configs/improved_experiment.yaml
```

当前默认只保留 4 个 SLERP 合并模型：

| 模型 | t | 设计目的 |
|---|---:|---|
| `slerp_code090_math010` | 0.10 | 极偏 Code，优先争取 HumanEval 明显提升 |
| `slerp_code080_math020` | 0.20 | 偏 Code，主力候选，兼顾少量 Math 注入 |
| `slerp_code070_math030` | 0.30 | 中等偏 Code，寻找能力冲突拐点 |
| `slerp_code020_math080` | 0.80 | 已有最佳双 benchmark 候选，作为 Math-heavy 锚点 |

第一轮 improved 结果表明：

- `expert_math`: GSM8K 0.77，高于 base 0.02。
- `expert_code`: HumanEval pass@1 0.60，高于 base 0.00。
- `slerp_code020_math080`: GSM8K 0.21，HumanEval pass@1 0.02，是第一轮里唯一两个 benchmark 都高于 base 的合并模型。
- Task Arithmetic 与 TIES 在已测配置下 HumanEval pass@1 均为 0.00，GSM8K 也接近 0，因此不再放入默认最终流水线。

新的搜索方向是补齐第一轮没有覆盖的 code-heavy SLERP 区间。因为 `expert_code` 本身在 GSM8K 上也达到 0.55，偏 Code 的 SLERP 有机会比 math-heavy 合并模型更好地保留 HumanEval，同时仍显著高于 base 的 GSM8K。

## 推荐运行方式

从项目根目录运行：

```powershell
cd D:\IntroML\project\level1_model_merging
.\scripts\run_improved_pipeline.ps1 -LimitMath 100 -LimitCode 50 -SkipExisting
```

如果只想先跑一个最可能提升 HumanEval 的候选：

```powershell
.\scripts\run_improved_pipeline.ps1 -Only slerp_code080_math020 -LimitMath 100 -LimitCode 50 -SkipExisting
```

如果中断后只想继续评测已生成的模型：

```powershell
.\scripts\run_improved_eval_resumable.ps1 -LimitMath 100 -LimitCode 50
```

## 结果文件

默认输出仍写入：

```text
results_improved/
  eval/
  plots/
  results_summary.csv
```

`summarize.py` 现在默认只汇总当前配置里列出的模型，避免历史失败模型污染最终 CSV。若报告需要引用第一轮 Task/TIES 消融结果，可以显式加入历史结果：

```powershell
conda run -n IntroML python -m model_merging_level1.summarize --config configs/improved_experiment.yaml --include-stale
```

## 可选磁盘清理

旧的无效合并模型目录不会被默认流水线使用。如果需要释放磁盘，可以先预览清理动作：

```powershell
.\scripts\cleanup_ineffective_improved_models.ps1 -WhatIf
```

确认不再需要这些历史模型后再执行真实清理：

```powershell
.\scripts\cleanup_ineffective_improved_models.ps1
```

建议在最终报告定稿后再清理，这样可以保留历史消融证据。
