# Level 1 实验报告模板：模型合并与知识复用

## 1. 实验目标

本实验使用 MergeKit 将 Code 专家模型与 Math 专家模型进行权重合并，验证合并后的模型是否能同时保留代码生成能力和数学推理能力。

## 2. 模型选择

| 角色 | 模型 |
|---|---|
| 基座模型 | `Qwen/Qwen2.5-1.5B` |
| Code 专家 | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Math 专家 | `Qwen/Qwen2.5-Math-1.5B-Instruct` |

选择理由：

- 三个模型均属于 Qwen2.5 1.5B 系列，参数规模接近，适合本地课程实验。
- Code 与 Math 专家具有不同能力方向，适合验证模型合并中的能力复用。
- 1.5B 参数规模适合 8GB 显存本地运行。
- 使用非指令微调的 `Qwen/Qwen2.5-1.5B` 作为 base，可以更清楚地展示专家模型和合并模型相对基础模型的提升。

## 3. 合并方法

| 方法 | 配置 | 说明 |
|---|---|---|
| SLERP | `slerp_code090_math010` / `slerp_code080_math020` / `slerp_code070_math030` / `slerp_code020_math080` | 在两个专家权重之间做球面插值；最终快速配置重点搜索 code-heavy 区间 |
| Task Arithmetic | 第一轮消融：`task_code025_math025_l050` / `task_code035_math035_l060` | 以基座模型为原点叠加两个任务向量；当前配置下效果较差，不放入最终快速流水线 |
| TIES | 第一轮消融：`ties_code025_math025_d020` / `ties_code035_math035_d030` | 通过裁剪、符号选择和稀疏化降低任务冲突；当前配置下效果较差，不放入最终快速流水线 |

## 4. Benchmark

| 能力 | Benchmark | 指标 |
|---|---|---|
| 数学推理 | GSM8K | Accuracy |
| 代码生成 | HumanEval | pass@1 |

HumanEval pass@1 通过执行模型生成代码得到。评测脚本会保存每道题的 `program` 和 `execution_output`，便于检查失败样例。

## 5. 结果

将 `results_improved/results_summary.csv` 中的结果填入下表：

| 模型 | GSM8K | HumanEval |
|---|---:|---:|
| Base |  |  |
| Code Expert |  |  |
| Math Expert |  |  |
| `slerp_code090_math010` |  |  |
| `slerp_code080_math020` |  |  |
| `slerp_code070_math030` |  |  |
| `slerp_code020_math080` |  |  |

建议插入：

- `results_improved/plots/bar_accuracy.png`
- `results_improved/plots/radar_accuracy.png`
- `results_improved/plots/slerp_weight_curve.png`

## 6. 分析

可以从以下角度讨论：

- 合并模型是否在两个任务上均高于基座模型。
- Code/Math 专家能力是否存在冲突。
- SLERP 插值比例变化是否呈现能力迁移趋势。
- Task Arithmetic 与 TIES 在第一轮消融中为什么不如 SLERP 稳。
- 若某个方法退化，可能原因包括基座差异、任务向量冲突、评测样本过少或生成参数不合适。

## 7. 结论

总结最优合并方法、最优权重比例，以及该结果对“无需重新训练即可复用多模型知识”的启示。
