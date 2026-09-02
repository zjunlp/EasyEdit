# Qwen3.8-27B examples

Scripts in this folder **do not kill or preempt** other GPU jobs. Weight-loading
jobs pick the emptiest card and **exit if free memory is below 66GB**.

Default yaml `model_name` is `./hugging_cache/Qwen3.8-27B`. Override with `--model`.

Environment: `transformers>=5.8` (see `requirements-qwen38.txt`). Do not bump
the repo-root `requirements.txt` (`transformers==5.5.4`).

## CPU (safe while other GPUs are busy)

```bash
python examples/qwen38/00_check_env.py
python examples/qwen38/run_cpu_checks.py
```

`run_cpu_checks.py` should print `[done] CPU checks passed`.

## GPU (only when a card has >= 66GB free)

```bash
MODEL=./hugging_cache/Qwen3.8-27B   # or your local checkpoint path

python examples/qwen38/01_smoke_load.py --model $MODEL
python examples/qwen38/02_explore_modules.py --model $MODEL
python examples/qwen38/03_wise_edit.py \
  --hparams hparams/WISE/qwen3.8-27b.yaml \
  --model $MODEL
python examples/qwen38/04_layer_scan.py \
  --layers 45,49,51,53,55,57,59,61,63 \
  --model $MODEL
# FT yaml is UNTESTED (structural smoke only):
python examples/qwen38/05_ft_edit.py --hparams hparams/FT/qwen3.8-27b.yaml --model $MODEL
```

Results are written to `examples/qwen38/results/`.

## Acceptance

| Script | Expect |
| --- | --- |
| `01_smoke_load.py` | loader succeeds; 64× `model.language_model.layers.*.mlp.down_proj` |
| `03_wise_edit.py` | free generation: France capital → Shanghai; Japan stays Tokyo. Teacher-forcing `rewrite_acc` may stay 0 on this hybrid-attention model; use `vanilla_metrics` in the JSON. |
| `run_cpu_checks.py` | unit tests pass; existing `qwen2.5-7b.yaml` still loads |

ROME is **not** claimed to work. See `hparams/ROME/qwen3.8-27b.yaml.example`.

Lab notes (Chinese, local paths): [docs/qwen3.8-27b.zh.md](../../docs/qwen3.8-27b.zh.md).
