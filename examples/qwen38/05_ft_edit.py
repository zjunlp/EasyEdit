#!/usr/bin/env python
"""FT single-fact smoke on Qwen3.8-27B (UNTESTED quality).

Same GPU guard as the WISE scripts. Yaml is experimental: single MLP layer only.

Usage:
    python examples/qwen38/05_ft_edit.py \\
        --hparams hparams/FT/qwen3.8-27b.yaml \\
        --model ./hugging_cache/Qwen3.8-27B
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

DEFAULT_HPARAMS = ROOT / "hparams" / "FT" / "qwen3.8-27b.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--target", default="Shanghai")
    parser.add_argument("--ground-truth", default="Paris")
    parser.add_argument("--subject", default="France")
    parser.add_argument("--loc-prompt", default="The capital of Japan is")
    parser.add_argument("--loc-gt", default="Tokyo")
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    return parser.parse_args()


def ask(tok, gen_model, question, max_new=16, chat=True):
    import torch

    if chat:
        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt = question
    inputs = tok(prompt, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def main() -> int:
    args = parse_args()
    gpu_idx, gpu_free_gb = pick_gpu(args.min_free_gb)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch  # noqa: E402
    from easyeditor import BaseEditor  # noqa: E402
    from easyeditor.models.ft.ft_hparams import FTHyperParams  # noqa: E402

    print(f"[gpu ] selected GPU{gpu_idx} (free {gpu_free_gb:.1f}GB)")
    print("[note] FT yaml is UNTESTED on Qwen3.8-27B; this is a structural smoke")

    hparams = FTHyperParams.from_hparams(args.hparams)
    if args.model:
        hparams.model_name = args.model
    hparams.device = 0
    if args.layer is not None:
        hparams.layers = [args.layer]

    t0 = time.time()
    editor = BaseEditor.from_hparams(hparams)
    torch.cuda.reset_peak_memory_stats()
    metrics, edited_model, _ = editor.edit(
        prompts=[args.prompt],
        ground_truth=[args.ground_truth],
        target_new=[args.target],
        subject=[args.subject],
        locality_inputs={
            "neighborhood": {
                "prompt": [args.loc_prompt],
                "ground_truth": [args.loc_gt],
            }
        },
        keep_original_weight=False,
        sequential_edit=True,
    )
    edit_s = time.time() - t0
    gen_model = getattr(edited_model, "model", edited_model)
    if not hasattr(gen_model, "generate"):
        gen_model = edited_model
    tok = editor.tok
    answers = {
        f"[chat] {args.prompt}": ask(tok, gen_model, args.prompt),
        f"[chat] {args.loc_prompt}": ask(tok, gen_model, args.loc_prompt),
    }
    report = {
        "untested": True,
        "model_path": hparams.model_name,
        "layers": hparams.layers,
        "metrics": metrics[0] if metrics else {},
        "post_edit_answers": answers,
        "edit_seconds": round(edit_s, 1),
        "peak_vram_GB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "gpu_index": gpu_idx,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"05_ft_edit{tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(answers, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
