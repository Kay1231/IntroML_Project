# Official KnOTS Table 1 Subset Run

This folder contains a small official-repo reproduction attempt for the KnOTS paper Table 1 setting.

## Scope

- Official code: `D:\IntroML\project\level2_knots_official`
- Model family: ViT-B/32 CLIP
- Experts: all 8 official rank-16 LoRA experts from `hoffman-lab`
- Evaluated dataset completed locally: MNIST
- Planned but blocked datasets: SVHN and GTSRB

SVHN and GTSRB were not completed because their official dataset hosts were blocked by the local Windows socket permission:

```text
以一种访问权限不允许的方式做了一个访问套接字的尝试
```

The scripts and configs still support `--datasets mnist svhn gtsrb`; once the datasets are manually placed under `level2_knots_official\data`, the same runner can evaluate them.

## Compatibility Fixes

The official code needed small compatibility fixes for the current local environment:

- Use the cached local `openai/clip-vit-base-patch32` snapshot with a hard-linked `model.safetensors`, avoiding repeated online safetensors conversion checks.
- Normalize PEFT state_dict keys when writing merged updates back into the model, including `.base_model.model.` and `.base_layer.weight` variants.
- Replace an interactive `pdb.set_trace()` error path with a clear `RuntimeError`.
- Add `--datasets` filtering in `eval_scripts/8vision_pertask_subset.py`.

## Results

The official KnOTS paper Table 1 reports normalized accuracy over 8 vision datasets. The relevant official MNIST values are:

| method | official MNIST normalized accuracy |
|---|---:|
| TIES | 56.8 |
| KnOTS-TIES | 68.9 |

The local MNIST subset run produced:

| method | config | dataset | accuracy | normalized accuracy |
|---|---|---:|---:|---:|
| KnOTS-TIES | `vitB_r16_knots_ties_subset` | MNIST | 69.10 | 69.587 |
| TIES | `vitB_r16_ties_subset` | MNIST | 56.99 | 57.392 |

Direct comparison:

| method | official MNIST normalized accuracy | local MNIST accuracy | local MNIST normalized accuracy | local - official |
|---|---:|---:|---:|---:|
| TIES | 56.8 | 56.99 | 57.392 | +0.592 |
| KnOTS-TIES | 68.9 | 69.10 | 69.587 | +0.687 |

Result files:

```text
D:\IntroML\project\level2_knots_official\results_subset\vitB_r16_knots_ties_subset_mnist_results.csv
D:\IntroML\project\level2_knots_official\results_subset\vitB_r16_ties_subset_mnist_results.csv
D:\IntroML\project\level2_knots_official\results_subset\table1_subset_summary.csv
D:\IntroML\project\level2_knots_official\results_subset\table1_subset_official_comparison.csv
```

## Interpretation

On the completed official MNIST subset, KnOTS-TIES is clearly better than direct TIES under the same 8-expert merge setting: 69.10% vs 56.99% accuracy. This supports the core claim that SVD-based alignment before TIES can reduce conflicts between task updates.

This is not a full absolute reproduction of Table 1. It is a small official-repo subset run used as supplementary evidence alongside the project-level Qwen LoRA KnOTS reproduction.
