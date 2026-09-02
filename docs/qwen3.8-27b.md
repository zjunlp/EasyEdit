# Qwen3.8-27B on EasyEdit

Qwen3.8-27B is **not** the same architecture as pure-text Qwen3.5-9B. It is a VL-wrapped language model:

| Field | Qwen3.8-27B | Qwen3.5-9B |
| --- | --- | --- |
| `architectures` | `Qwen3_5ForConditionalGeneration` | CausalLM (`Qwen3_5ForCausalLM` / Qwen3) |
| `model_type` | `qwen3_5` with `vision_config` + `text_config` | text-only |
| Backbone | 64 hybrid-attention layers (48 GDN `linear_attn` + 16 `self_attn`) | standard full attention |
| Edit module path | `model.language_model.layers.{}.mlp.down_proj` | `model.layers.{}.mlp.down_proj` |
| Path substring `vl` | **absent** — `vl_utils.py` aliases do not fire | n/a |

Detection therefore reads `AutoConfig` (`architectures` / nested vision+text configs). A `qwen3.8` path hint is auxiliary only and is **not** used as the sole criterion, so Qwen3.5-9B still uses `AutoModelForCausalLM`.

```
hparams yaml → BaseEditor.__init__
                 ├─ is_qwen35_vl_text_model?  → model_loader.py
                 │     AutoModelForImageTextToText + language_model_only=True, bfloat16, eager attn
                 └─ name contains qwen2/qwen3 → existing CausalLM branch (Qwen3.5-9B)
```

The loader fills `model.config.hidden_act` from `text_config` (WISE activation distance needs it). It does **not** set tokenizer `padding_side`; the editor still chooses padding per algorithm.

Official yaml `model_name` is `./hugging_cache/Qwen3.8-27B`. Override with `--model /path/to/Qwen3.8-27B`.

## Algorithm compatibility

| Priority | Algorithm | Structure | This work | Risk |
| --- | --- | --- | --- | --- |
| P0 verified | **WISE** | `inner_params` same prefix as `qwen3vl-4b.yaml` | yaml + smoke load; padding/thinking fixes for generation flip | default left pad + thinking template inflate ES without flipping free gen |
| P1 runnable | **FT** | `rewrite_module_tmp` with `language_model` prefix | experimental yaml, **untested** | 27B full FT OOM; yaml edits one MLP layer |
| P1 runnable | **GRACE** | same `inner_params` as WISE | experimental yaml, **untested** | not measured on 27B |
| P1 | **IKE** | no weight change | load success is enough to try | not a current target |
| P1 module names | **LoRA** | `q_proj` / `v_proj` only on full_attention layers | documented, no yaml | GDN layers lack those modules |
| P2 layer pick | **ROME / R-ROME / MEMIT / AlphaEdit** | nethook works; causal trace / mom2 on GDN unverified | `04_layer_scan.py` + ROME yaml.example | `attn_module_tmp` is `self_attn` vs `linear_attn`; needs mom2 stats |
| Later | MEND / SERAC / UltraEdit / KN / QLoRA | pretrained editor or architecture-sensitive tracing | listed as follow-up | 27B cost |

Default editor dtype is **fp32**; 27B fp32 OOMs. The Qwen3.5 VL-text loader **forces bfloat16**. Attention uses `eager` by default (SDPA + gradient checkpointing hits a non-contiguous 4D mask error).

## Environment

Do **not** change repo-root `requirements.txt` (`transformers==5.5.4`). Use:

```bash
# transformers>=5.8 — see requirements-qwen38.txt
python examples/qwen38/00_check_env.py
```

`00_check_env.py` is read-only (nvidia-smi query + version print). It does not load 27B weights.

VRAM floor for weight-loading scripts: **66GB free** on the chosen GPU (52GB weights + 8GB edit/generate + 6GB margin). Scripts pick the emptiest card and **refuse** below that floor so they do not steal occupied GPUs.

## Commands (run only when a card is free)

See [examples/qwen38/README.md](../examples/qwen38/README.md) for the full script list.

```bash
python examples/qwen38/00_check_env.py
python examples/qwen38/run_cpu_checks.py

MODEL=./hugging_cache/Qwen3.8-27B
python examples/qwen38/01_smoke_load.py --model $MODEL
python examples/qwen38/02_explore_modules.py --model $MODEL
python examples/qwen38/03_wise_edit.py \
  --hparams hparams/WISE/qwen3.8-27b.yaml \
  --model $MODEL
python examples/qwen38/04_layer_scan.py \
  --layers 45,49,51,53,55,57,59,61,63 \
  --model $MODEL
python examples/qwen38/05_ft_edit.py --model $MODEL
```

Results go under `examples/qwen38/results/`. GPU scripts refuse to start if the emptiest card has less than 66GB free.

## Verified behavior

**Loader:** the model loads through `BaseEditor` / `model_loader.py`, chat generation works, MLP paths are `model.language_model.layers.*.mlp.down_proj`.

**WISE:** locates the prompt/target cut from the first non-`-100` label (works with right padding), `model.train()` during the edit loop so transformers 5.8 gradient checkpointing actually runs, optional hparams `padding_side` / `enable_thinking` / `attn_implementation` / `language_model_only`. The WISE yaml sets `padding_side: right` and `enable_thinking: false`. Verified: chat `"The capital of France is"` → `Shanghai`, Japan locality stays Tokyo. Peak ~53GB.

On this hybrid-attention model, EasyEdit teacher-forcing `rewrite_acc` can stay 0 even when free generation flips (eval temporarily uses left padding and requires the first token to be `Shanghai`). `examples/qwen38/03_wise_edit.py` therefore also writes `vanilla_metrics` (official `vanilla_generation` token match + substring checks).

## Known failure modes

1. **Left padding + GDN.** WISE `_cal_ft_loss` used `(labels==-100).sum()-1`. With left pad the mask can match the transition (ES=1.0) while GDN recurrent state is polluted by leading pads that generation does not have, so free generation does not flip.
2. **Right padding without loc fix.** Trailing `-100` pads are counted as prompt, so the mask lands on pad tokens and the model “learns pad”.
3. **Thinking template.** `apply_chat_template` defaults to a thinking block. Training then puts `" Shanghai"` in the thinking distribution; demo generation with `enable_thinking=False` does not see the edit.
4. **Gradient checkpointing in `eval()`.** transformers 5.8 `GradientCheckpointingLayer` requires `self.training`. Edit-time `model.eval()` silently skips GC → 27B OOM.
5. **Layer 29.** GDN layer at ~45% depth can move teacher-forcing metrics without transferring to free generation. Prefer layer **53** (yaml default). Scan full_attention 51/55/59/63 as well.

## ROME scaffold

- `examples/qwen38/04_layer_scan.py` — WISE layer scan including GDN 53 and full_attention 51/55/59/63.
- `hparams/ROME/qwen3.8-27b.yaml.example` — path template `model.language_model.layers.{}.mlp.down_proj`. Comments cover `self_attn` vs `linear_attn`. **Do not claim ROME works.** Causal tracing and mom2 on GDN recurrent state are unknown; promote to a real yaml only after rewrite+locality.
