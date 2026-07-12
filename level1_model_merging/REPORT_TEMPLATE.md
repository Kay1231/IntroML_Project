# Level 1 实验报告模板：模型合并与知识复用

## 1. 实验目标

本实验使用 MergeKit 将 Code 专家模型与 Math 专家模型进行权重合并，验证合并后的模型是否能同时保留代码生成能力和数学推理能力。

## 2. 模型选择

| 角色 | 模型 |
|---|---|
| 基座模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| Code 专家 | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Math 专家 | `Qwen/Qwen2.5-Math-1.5B-Instruct` |

选择理由：

- 三个模型均属于 Qwen2.5 1.5B 系列，结构与 tokenizer 兼容性较好。
- Code 与 Math 专家具有不同能力方向，适合验证模型合并中的能力复用。
- 1.5B 参数规模适合 8GB 显存本地运行。

## 3. 合并方法

| 方法 | 配置 | 说明 |
|---|---|---|
| SLERP | `slerp_code070_math030` / `slerp_code050_math050` / `slerp_code030_math070` | 在两个专家权重之间做球面插值 |
| Task Arithmetic | `task_code050_math050` | 以基座模型为原点叠加两个任务向量 |
| TIES | `ties_code050_math050_d050` | 通过裁剪、符号选择和稀疏化降低任务冲突 |

## 4. Benchmark

| 能力 | Benchmark | 指标 |
|---|---|---|
| 数学推理 | GSM8K | Accuracy |
| 代码生成 | HumanEval | pass@1 或 Syntax Rate |

HumanEval pass@1 需要执行模型生成代码；若未启用 `-AllowCodeExecution`，报告中应说明使用的是 Syntax Rate 作为快速替代指标。

## 5. 结果

将 `results/results_summary.csv` 中的结果填入下表：

| 模型 | GSM8K | HumanEval |
|---|---:|---:|
| Base |  |  |
| Code Expert |  |  |
| Math Expert |  |  |
| SLERP 0.5 |  |  |
| Task Arithmetic |  |  |
| TIES |  |  |

建议插入：

- `results/plots/bar_accuracy.png`
- `results/plots/radar_accuracy.png`
- `results/plots/slerp_weight_curve.png`

## 6. 分析

可以从以下角度讨论：

- 合并模型是否在两个任务上均高于基座模型。
- Code/Math 专家能力是否存在冲突。
- SLERP 插值比例变化是否呈现能力迁移趋势。
- Task Arithmetic 与 TIES 是否比普通插值更稳。
- 若某个方法退化，可能原因包括基座差异、任务向量冲突、评测样本过少或生成参数不合适。

## 7. 结论

总结最优合并方法、最优权重比例，以及该结果对“无需重新训练即可复用多模型知识”的启示。
