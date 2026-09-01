#!/usr/bin/env python
"""Smoke-load Qwen3.8-27B through the EasyEdit VL-text loader.

Picks the GPU with the most free memory and refuses to run if that card has
less than 66GB free (52GB weights + 8GB generate headroom + 6GB floor).
Does not kill or preempt other processes.

Set CUDA_VISIBLE_DEVICES after the nvidia-smi pick so only the chosen card
is visible. Override the checkpoint with --model.

Usage:
    conda activate EasyEdit-next
    python examples/qwen38/01_smoke_load.py --model /data/zhangzuhao/models/Qwen3.8-27B

Results:
    examples/qwen38/results/01_smoke_load.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_FREE_GB = 66.0
DEFAULT_MODEL = "./hugging_cache/Qwen3.8-27B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Checkpoint path (official yaml uses ./hugging_cache/Qwen3.8-27B)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_GB,
        help="Refuse to run if the emptiest GPU is below this free-memory floor",
    )
    return parser.parse_args()


def pick_gpu(min_free_gb: float):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    best = None
    for line in out.strip().splitlines():
        idx, free_mib = [x.strip() for x in line.split(",")]
        free_gb = int(free_mib) / 1024
        if best is None or free_gb > best[1]:
            best = (int(idx), free_gb)
    if best is None:
        raise RuntimeError("nvidia-smi reported no GPUs")
    if best[1] < min_free_gb:
        raise RuntimeError(
            f"largest free GPU{best[0]} has only {best[1]:.1f}GB < {min_free_gb:.1f}GB; "
            "refusing to run so existing jobs are not disturbed"
        )
    return best


def main() -> int:
    args = parse_args()
    gpu_idx, gpu_free_gb = pick_gpu(args.min_free_gb)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(ROOT))

    import torch  # noqa: E402
    import transformers  # noqa: E402
    from easyeditor.util.model_loader import (  # noqa: E402
        is_qwen35_vl_text_model,
        load_qwen35_language_model,
    )

    print(
        f"[env ] transformers={transformers.__version__}  torch={torch.__version__}"
    )
    print(
        f"[gpu ] selected GPU{gpu_idx} (free {gpu_free_gb:.1f}GB, floor {args.min_free_gb:.1f}GB); "
        f"CUDA_VISIBLE_DEVICES={gpu_idx}"
    )
    if not is_qwen35_vl_text_model(args.model):
        raise SystemExit(
            f"{args.model} was not detected as Qwen3.5 VL-text "
            "(Qwen3_5ForConditionalGeneration). Pure-text Qwen3.5-9B should use CausalLM."
        )

    t0 = time.time()
    # device=0 is the selected physical GPU after CUDA_VISIBLE_DEVICES remap.
    model, tok = load_qwen35_language_model(
        args.model, device=0, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    load_s = time.time() - t0
    print(f"[load] EasyEdit loader ok ({load_s:.0f}s)")

    names = [n for n, _ in model.named_modules()]
    has_vision = any("vision" in n or "visual" in n for n in names)
    mlp_layers = [
        n for n in names if "language_model" in n and n.endswith("mlp.down_proj")
    ]
    print(
        f"[arch] vision/visual modules present={has_vision}; "
        f"language_model MLP count={len(mlp_layers)}"
    )
    if mlp_layers:
        print(f"[arch] MLP path sample: {mlp_layers[0]}  /  {mlp_layers[-1]}")

    model.eval()
    try:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen_s = time.time() - t1
    new_tokens = tok.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"[gen ] {new_tokens!r}  ({gen_s:.1f}s)")

    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(
        f"[mem ] peak {peak_gb:.1f}GB / free-before {gpu_free_gb:.1f}GB; "
        f"params {n_params:.1f}B"
    )

    report = {
        "model_path": args.model,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "load_method": "easyeditor.util.model_loader.load_qwen35_language_model",
        "detected_qwen35_vl_text": True,
        "load_seconds": round(load_s, 1),
        "num_params_B": round(n_params, 2),
        "has_vision_modules": has_vision,
        "language_model_mlp_layer_count": len(mlp_layers),
        "mlp_path_example": mlp_layers[0] if mlp_layers else None,
        "mlp_path_last": mlp_layers[-1] if mlp_layers else None,
        "prompt_tail": prompt[-80:],
        "generated": new_tokens,
        "gen_seconds": round(gen_s, 1),
        "peak_vram_GB": round(peak_gb, 1),
        "gpu_index": gpu_idx,
        "gpu_free_GB_before": round(gpu_free_gb, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "01_smoke_load.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
