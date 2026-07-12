# Level 1 改进说明

这套改动的目标是让实验结果更接近课程要求：expert 在各自 benchmark 上应当优于 base，并尽量找到一个 merged model 在 GSM8K 和 HumanEval 上都优于 base。

## 改了什么

1. `evaluate.py`
   - 修复 HumanEval 解析 bug：原代码会把 ` ```python ... ``` ` 里的代码丢掉，导致评测不是在真实生成代码上进行。
   - HumanEval 默认可计算 `pass_at_1`：运行测试用例后才算代码正确，而不是只看语法是否通过。
   - 保存 `program` 字段，方便检查每一道题最终执行的完整 Python 程序。
   - 改进 GSM8K prompt：禁止 Python/伪代码，要求最后一行输出 `#### <number>`。
   - 改进 GSM8K 答案抽取：优先读 `####`、`Final answer`、`\boxed{}`，兜底时会忽略代码块，减少抽到中间变量的概率。
   - 增加 `--skip-existing`，避免长时间实验被重复计算。

2. `improved_experiment.yaml`
   - 把 base 改为 `Qwen/Qwen2.5-1.5B`，用真正的基础模型作为“微调前 baseline”。
   - 保留 code/math expert。
   - 新增更保守的 Task Arithmetic / TIES 权重，降低两个 expert 直接相互干扰的风险。
   - 新结果写到 `generated_merge_configs_improved`、`merged_models_improved`、`results_improved`，不会覆盖旧结果。

3. `run_improved_pipeline.ps1`
   - 一键生成配置、合并、评测、汇总。
   - 默认启用 HumanEval 代码执行，得到真正的 pass@1。

## 预计资源

你已经下载过两个 expert 的大部分权重。新增的 `Qwen/Qwen2.5-1.5B` base 大约需要 3GB 磁盘空间。改进配置里有 7 个 merged model，每个约 3GB 左右，最坏情况额外需要约 24GB 到 28GB 磁盘空间，加上缓存建议 D 盘至少预留 35GB。

RTX 4060 Laptop 8GB 可以跑，但评测会比较慢。`LimitMath 100` + `LimitCode 50` 是合适的快速实验规模；最终报告如果时间允许，可以把 limit 调大。

## 建议运行方式

从项目根目录运行：

```powershell
cd D:\IntroML\project\level1_model_merging
.\scripts\run_improved_pipeline.ps1 -LimitMath 100 -LimitCode 50
```

如果中途失败，修好后可以加 `-SkipExisting` 续跑：

```powershell
.\scripts\run_improved_pipeline.ps1 -LimitMath 100 -LimitCode 50 -SkipExisting
```

如果磁盘空间不够，先只跑最可能成功的保守 task-vector merge：

```powershell
.\scripts\run_improved_pipeline.ps1 -Only task_code025_math025_l050 -LimitMath 100 -LimitCode 50
```
