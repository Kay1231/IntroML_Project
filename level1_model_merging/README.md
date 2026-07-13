# Level 1 Model Merging 实验工程

这个目录实现课程项目 Level 1：把两个同系列、不同专长的专家模型进行权重合并，并在数学与代码 benchmark 上比较基座模型、源专家模型和合并模型。

当前推荐实验选择：

- 基座模型：`Qwen/Qwen2.5-1.5B`
- Code 专家：`Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Math 专家：`Qwen/Qwen2.5-Math-1.5B-Instruct`
- 默认最终合并算法：code-heavy `SLERP` 小网格
- 历史消融算法：`Task Arithmetic`、`TIES`
- 数学评测：`GSM8K`
- 代码评测：`HumanEval pass@1`

选择 1.5B 模型是为了适配当前电脑的 RTX 4060 Laptop GPU 8GB 显存；默认 batch size 为 1，适合本机小规模课程实验。

## 目录结构

```text
level1_model_merging/
  configs/default_experiment.yaml        # 初始实验配置
  configs/improved_experiment.yaml       # 当前推荐最终配置
  scripts/                               # Windows PowerShell 入口脚本
  src/model_merging_level1/              # Python 代码
  requirements.txt                       # pip 依赖
  environment.yml                        # conda 环境参考
  REPORT_TEMPLATE.md                     # 实验报告模板
```

## 资源与数据大小

当前机器检测到：

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，显存约 8GB
- C 盘可用空间约 107GB，D 盘可用空间约 166GB

建议配置：

- GPU：8GB 显存可以跑默认 1.5B 方案；如果 OOM，减少 `max_new_tokens` 或只评测部分模型。
- 内存：建议 16GB 起步，32GB 更稳。
- 磁盘：建议至少预留 35GB。

下载量估计：

- 三个源模型/基座模型缓存：每个约 3GB，合计约 9-10GB。
- 默认 4 个合并结果：每个约 3GB，合计约 12GB。
- GSM8K 和 HumanEval 数据集很小，通常只占几十 MB 以内；真正占空间的是模型权重。
- PyTorch/CUDA 与 Python 依赖可能额外占 3-6GB。

## 安装依赖

项目要求使用你已经创建的 `IntroML` 环境。先进入本目录：

```powershell
cd D:\IntroML\project\level1_model_merging
```

如果 `IntroML` 里还没有 GPU 版 PyTorch，先执行：

```powershell
conda install -n IntroML -y pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

然后安装其余依赖：

```powershell
.\scripts\setup_intro_ml.ps1
```

默认 Hugging Face 模型与数据缓存会写到 `D:\IntroML\project\hf_cache`，避免占用 C 盘。若网络下载 Hugging Face 模型较慢，可以设置国内镜像或提前登录：

```powershell
huggingface-cli login
```

## 推荐运行实验

当前建议直接使用改进版一键流水线：

```powershell
.\scripts\run_improved_pipeline.ps1 -LimitMath 100 -LimitCode 50 -SkipExisting
```

如果只想先跑一个 code-heavy 候选：

```powershell
.\scripts\run_improved_pipeline.ps1 -Only slerp_code080_math020 -LimitMath 100 -LimitCode 50 -SkipExisting
```

如果评测中断，可以使用逐模型逐 benchmark 的断点脚本：

```powershell
.\scripts\run_improved_eval_resumable.ps1 -LimitMath 100 -LimitCode 50
```

旧版分步脚本仍然保留：

```powershell
.\scripts\generate_merge_configs.ps1
.\scripts\run_merges.ps1
.\scripts\evaluate_all.ps1 -LimitMath 100 -LimitCode 50 -AllowCodeExecution -SkipExisting
.\scripts\summarize_results.ps1
```

结果会写到：

```text
results_improved/
  eval/
  plots/
  results_summary.csv
```

## 建议报告写法

报告可以围绕这条主线展开：

1. 基座模型在 GSM8K/HumanEval 上作为下界。
2. Code 专家应在 HumanEval 上强于基座，Math 专家应在 GSM8K 上强于基座。
3. 合并模型若在两个任务上都高于基座，说明保留了双侧能力。
4. 比较 SLERP 不同插值比例、Task Arithmetic 和 TIES 的差异。
5. 用 `results_improved/plots/` 下的柱状图、雷达图、权重曲线支撑结论。

`REPORT_TEMPLATE.md` 里已经放了可直接填写的报告模板。
