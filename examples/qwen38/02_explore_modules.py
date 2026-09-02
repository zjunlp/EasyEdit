#!/usr/bin/env python
"""S1 structure probe: layer types and module-path templates for yaml.

Loads through EasyEdit's VL-text loader. Refuses to run if free GPU < 66GB.

Usage:
    python examples/qwen38/02_explore_modules.py --model ./hugging_cache/Qwen3.8-27B

Results:
    examples/qwen38/results/02_modules.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from gpu_guard import MIN_FREE_GB, pick_gpu  # noqa: E402

DEFAULT_MODEL = "./hugging_cache/Qwen3.8-27B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_idx, gpu_free_gb = pick_gpu(args.min_free_gb)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch  # noqa: E402
    from easyeditor.util.model_loader import load_qwen35_language_model  # noqa: E402

    print(f"[gpu ] selected GPU{gpu_idx} (free {gpu_free_gb:.1f}GB)")
    t0 = time.time()
    model, tok = load_qwen35_language_model(
        args.model, device=0, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    print(f"[load] {time.time() - t0:.0f}s")

    text_cfg = getattr(model.config, "text_config", model.config)
    layer_types = list(getattr(text_cfg, "layer_types", []) or [])
    full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    lin_idx = [i for i, t in enumerate(layer_types) if t == "linear_attention"]

    names = [n for n, _ in model.named_modules()]
    lm_names = [n for n in names if "language_model" in n]
    mlp_paths = sorted(n for n in lm_names if n.endswith("mlp.down_proj"))
    self_attn = [n for n in lm_names if n.endswith("self_attn")]
    lin_attn = [n for n in lm_names if "linear_attn" in n]
    ln_f = next((n for n in names if n.endswith("language_model.norm")), None)
    lm_head = next((n for n in names if n == "lm_head"), None)

    template = None
    if mlp_paths:
        template = mlp_paths[0].replace(".0.", ".{}.")

    template_check = {}
    for flag in (None, False, True):
        kw = {} if flag is None else {"enable_thinking": flag}
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                tokenize=False,
                add_generation_prompt=True,
                **kw,
            )
            template_check[str(flag)] = {"ok": True, "tail": text[-60:]}
        except Exception as exc:
            template_check[str(flag)] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:120]}",
            }

    report = {
        "model_path": args.model,
        "num_layers": len(layer_types),
        "full_attention_layers": full_idx,
        "linear_attention_layers": lin_idx,
        "mlp_down_proj": {
            "count": len(mlp_paths),
            "first": mlp_paths[0] if mlp_paths else None,
            "last": mlp_paths[-1] if mlp_paths else None,
            "template": template,
        },
        "self_attn": {"count": len(self_attn), "first": self_attn[0] if self_attn else None},
        "linear_attn": {"count": len(lin_attn), "sample": lin_attn[:3]},
        "ln_f": ln_f,
        "lm_head": lm_head,
        "chat_template_flags": template_check,
        "gpu_index": gpu_idx,
        "gpu_free_GB_before": round(gpu_free_gb, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "02_modules.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[layers] {len(layer_types)} layers; full_attention={full_idx}")
    print(f"[mlp   ] template={template}")
    print(f"[done ] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
