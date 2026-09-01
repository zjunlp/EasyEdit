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

Official yaml `model_name` is `./hugging_cache/Qwen3.8-27B`. Override on this machine with `--model /data/zhangzuhao/models/Qwen3.8-27B`.

## Algorithm compatibility

| Priority | Algorithm | Structure | This work | Risk |
| --- | --- | --- | --- | --- |
| P0 verified | **WISE** | `inner_params` same prefix as `qwen3vl-4b.yaml` | PR-1 yaml + smoke load; PR-2 algorithm fixes for generation flip | default left pad + thinking template inflate ES without flipping free gen |
| P1 runnable | **FT** | `rewrite_module_tmp` with `language_model` prefix | experimental yaml, **untested** | 27B full FT OOM; yaml edits one MLP layer |
| P1 runnable | **GRACE** | same `inner_params` as WISE | experimental yaml, **untested** | not measured on 27B |
| P1 | **IKE** | no weight change | load success is enough to try | not a current target |
| P1 module names | **LoRA** | `q_proj` / `v_proj` only on full_attention layers | documented, no yaml | GDN layers lack those modules |
| P2 layer pick | **ROME / R-ROME / MEMIT / AlphaEdit** | nethook works; causal trace / mom2 on GDN unverified | PR-3 scaffold + `04_layer_scan.py` | `attn_module_tmp` is `self_attn` vs `linear_attn`; needs mom2 stats |
| Later | MEND / SERAC / UltraEdit / KN / QLoRA | pretrained editor or architecture-sensitive tracing | listed as follow-up | 27B cost |

Default editor dtype is **fp32**; 27B fp32 OOMs. The Qwen3.5 VL-text loader **forces bfloat16**. Attention uses `eager` by default (SDPA + gradient checkpointing hits a non-contiguous 4D mask error).

## Environment

Do **not** change repo-root `requirements.txt` (`transformers==5.5.4`). Use:

```bash
conda activate EasyEdit-next
# torch==2.9.1, transformers>=5.8.1 — see requirements-qwen38.txt
cd /data/zhangzuhao/ideaTest/EasyEdit
python examples/qwen38/00_check_env.py
```

`00_check_env.py` is read-only (nvidia-smi query + version print). It does not load 27B weights.

VRAM floor for weight-loading scripts: **66GB free** on the chosen GPU (52GB weights + 8GB edit/generate + 6GB margin). Scripts pick the emptiest card and **refuse** below that floor so they do not steal occupied GPUs.

## Commands (run only when a card is free)

```bash
conda activate EasyEdit-next
cd /data/zhangzuhao/ideaTest/EasyEdit

# PR-1: load + 8-token greedy chat generate + module-path sample
python examples/qwen38/01_smoke_load.py --model /data/zhangzuhao/models/Qwen3.8-27B

# PR-2 (feat/wise-ft-loss-padding): France → Shanghai generation flip
python examples/qwen38/03_wise_edit.py \
  --hparams hparams/WISE/qwen3.8-27b.yaml \
  --model /data/zhangzuhao/models/Qwen3.8-27B

# Optional VRAM knobs on the example only (not default WISE):
python examples/qwen38/03_wise_edit.py --small-context --model /data/zhangzuhao/models/Qwen3.8-27B

# PR-3 scaffold: GDN 53 and full_attention 51/55/59/63 (layer 29 fails free-gen)
python examples/qwen38/04_layer_scan.py \
  --layers 45,49,51,53,55,57,59,61,63 \
  --model /data/zhangzuhao/models/Qwen3.8-27B
```

Results go under `examples/qwen38/results/`. Do not run `01` / `03` / `04` while both A800s are occupied.

## PR-1 vs PR-2 acceptance

**PR-1** (`feat/qwen38-loader`): the model loads through `BaseEditor` / `model_loader.py`, chat generation works, MLP paths are `model.language_model.layers.*.mlp.down_proj`. WISE generation-flip is **not** a PR-1 criterion. `WISE.py` is not modified.

**PR-2** (`feat/wise-ft-loss-padding`): WISE locates the prompt/target cut from the first non-`-100` label (works with right padding), `model.train()` during the edit loop so transformers 5.8 gradient checkpointing actually runs, optional hparams `padding_side` / `enable_thinking` / `attn_implementation` / `language_model_only`. The WISE yaml now sets `padding_side: right` and `enable_thinking: false`. Verified target: chat `"The capital of France is"` → `Shanghai`, Japan locality stays Tokyo. Peak ~53GB.

## Known failure modes

1. **Left padding + GDN.** WISE `_cal_ft_loss` used `(labels==-100).sum()-1`. With left pad the mask can match the transition (ES=1.0) while GDN recurrent state is polluted by leading pads that generation does not have, so free generation does not flip.
2. **Right padding without loc fix.** Trailing `-100` pads are counted as prompt, so the mask lands on pad tokens and the model “learns pad”.
3. **Thinking template.** `apply_chat_template` defaults to a thinking block. Training then puts `" Shanghai"` in the thinking distribution; demo generation with `enable_thinking=False` does not see the edit.
4. **Gradient checkpointing in `eval()`.** transformers 5.8 `GradientCheckpointingLayer` requires `self.training`. Edit-time `model.eval()` silently skips GC → 27B OOM.
5. **Layer 29.** GDN layer at ~45% depth can move teacher-forcing metrics without transferring to free generation. Prefer layer **53** (yaml default). Scan full_attention 51/55/59/63 as well.

## PR-3 ROME scaffold (not a weekly PR)

- `examples/qwen38/04_layer_scan.py` — WISE layer scan including GDN 53 and full_attention 51/55/59/63.
- `hparams/ROME/qwen3.8-27b.yaml.example` — path template `model.language_model.layers.{}.mlp.down_proj`. Comments cover `self_attn` vs `linear_attn`. **Do not claim ROME works.** Causal tracing and mom2 on GDN recurrent state are unknown; promote to a real yaml only after rewrite+locality.
