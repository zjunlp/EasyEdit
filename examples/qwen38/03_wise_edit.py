#!/usr/bin/env python
"""WISE single-fact edit on Qwen3.8-27B via BaseEditor + yaml.

Right-padding loc, enable_thinking=False, and train-mode GC live in EasyEdit
itself. This script only: picks a free GPU, loads through BaseEditor,
optionally shrinks context templates for VRAM, then reports chat/raw generation
plus vanilla_metrics (teacher-forcing vs generate-match vs substring).

Usage:
    python examples/qwen38/03_wise_edit.py \\
        --hparams hparams/WISE/qwen3.8-27b.yaml \\
        --model ./hugging_cache/Qwen3.8-27B
    python examples/qwen38/03_wise_edit.py --small-context --tag mem

Results:
    examples/qwen38/results/03_wise_edit[_<tag>].json

Do not run while GPUs are occupied. Refuses to start if free < 66GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = Path(__file__).resolve().parent
RESULTS_DIR = EXAMPLES / "results"
sys.path.insert(0, str(EXAMPLES))
sys.path.insert(0, str(ROOT))

from gpu_guard import MIN_FREE_GB, pick_gpu  # noqa: E402

DEFAULT_HPARAMS = ROOT / "hparams" / "WISE" / "qwen3.8-27b.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument(
        "--model",
        default=None,
        help="Override yaml model_name (e.g. ./hugging_cache/Qwen3.8-27B)",
    )
    parser.add_argument("--layer", type=int, default=None, help="Override yaml inner_params layer")
    parser.add_argument("--act-ratio", type=float, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--target", default="Shanghai")
    parser.add_argument("--ground-truth", default="Paris")
    parser.add_argument("--subject", default="France")
    parser.add_argument("--loc-prompt", default="The capital of Japan is")
    parser.add_argument("--loc-gt", default="Tokyo")
    parser.add_argument(
        "--small-context",
        action="store_true",
        help="Example-only VRAM knob: cut WISE context templates 11→6. Not a default WISE change.",
    )
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    return parser.parse_args()


def ask(tok, gen_model, question: str, max_new: int = 16, chat: bool = True) -> str:
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
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def unwrap_generate_model(edited):
    gen_model = getattr(edited, "model", None)
    if gen_model is None or not hasattr(gen_model, "generate"):
        gen_model = edited
    return gen_model


def contains_token_span(text: str, needle: str) -> bool:
    return needle.lower() in (text or "").lower()


def official_vanilla_acc(model, tok, hparams, prompt, target, device):
    """EasyEdit GRACE-style generate-then-match (no teacher forcing, no left pad)."""
    from easyeditor.evaluate.evaluate_utils import test_prediction_acc

    return test_prediction_acc(
        model, tok, hparams, prompt, target, device, vanilla_generation=True
    )


def maybe_shrink_context_templates():
    import importlib

    wise_main = importlib.import_module("easyeditor.models.wise.wise_main")
    orig = wise_main.get_context_templates

    def _small(model_, tok_, length_params, device):
        return orig(model_, tok_, [[5, 5]], device)

    wise_main.get_context_templates = _small
    print("[mem ] --small-context: WISE templates [[5,5],[10,5]] → [[5,5]] (example only)")


def main() -> int:
    args = parse_args()
    gpu_idx, gpu_free_gb = pick_gpu(args.min_free_gb)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch  # noqa: E402
    from easyeditor import BaseEditor  # noqa: E402
    from easyeditor.models.wise.wise_hparams import WISEHyperParams  # noqa: E402

    print(
        f"[gpu ] selected GPU{gpu_idx} (free {gpu_free_gb:.1f}GB, floor {args.min_free_gb:.1f}GB); "
        f"CUDA_VISIBLE_DEVICES={gpu_idx}"
    )

    hparams = WISEHyperParams.from_hparams(args.hparams)
    if args.model:
        hparams.model_name = args.model
    hparams.device = 0
    if args.layer is not None:
        hparams.inner_params = [
            f"model.language_model.layers[{args.layer}].mlp.down_proj.weight"
        ]
    if args.act_ratio is not None:
        hparams.act_ratio = args.act_ratio
        print(f"[wise] act_ratio override {args.act_ratio}")
    print(
        f"[wise] inner_params={hparams.inner_params}  padding_side={hparams.padding_side}  "
        f"enable_thinking={hparams.enable_thinking}  edit_lr={hparams.edit_lr}  n_iter={hparams.n_iter}"
    )

    if args.small_context:
        maybe_shrink_context_templates()

    t_start = time.time()
    editor = BaseEditor.from_hparams(hparams)
    print(f"[load] BaseEditor ready ({time.time() - t_start:.0f}s)")

    if hasattr(editor.model, "gradient_checkpointing_enable"):
        editor.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(editor.model, "config"):
            editor.model.config.use_cache = False
        print("[mem ] gradient checkpointing enabled (edit-time train() is required for transformers 5.8)")

    torch.cuda.reset_peak_memory_stats()
    t_edit = time.time()
    metrics, edited_model, _ = editor.edit(
        prompts=[args.prompt],
        ground_truth=[args.ground_truth],
        target_new=[args.target],
        subject=[args.subject],
        loc_prompts=[args.subject],
        locality_inputs={
            "neighborhood": {
                "prompt": [args.loc_prompt],
                "ground_truth": [args.loc_gt],
            },
        },
        keep_original_weight=False,
        sequential_edit=True,
    )
    edit_seconds = time.time() - t_edit
    m = metrics[0] if metrics else {}
    print("[metrics] ===== edit =====")
    print(json.dumps(m, indent=2, ensure_ascii=False, default=str))

    gen_model = unwrap_generate_model(edited_model)
    try:
        for mod in (editor.model, gen_model):
            if hasattr(mod, "gradient_checkpointing_disable"):
                mod.gradient_checkpointing_disable()
            if hasattr(mod, "config"):
                mod.config.use_cache = True
    except Exception as exc:
        print(f"[gen ] restore use_cache failed (ignored): {exc}")

    tok = editor.tok
    post_edit_answers = {
        f"[chat] {args.prompt}": ask(tok, gen_model, args.prompt),
        f"[raw ] {args.prompt}": ask(tok, gen_model, args.prompt, chat=False),
        f"[chat] {args.loc_prompt}": ask(tok, gen_model, args.loc_prompt),
        f"[raw ] {args.loc_prompt}": ask(tok, gen_model, args.loc_prompt, chat=False),
        "[chat] What language do people speak in France?": ask(
            tok, gen_model, "What language do people speak in France?", 12
        ),
    }
    print("\n[gen ] ===== post-edit generation =====")
    for question, answer in post_edit_answers.items():
        print(f"  Q: {question}\n  A: {answer}\n")

    chat_rewrite = post_edit_answers[f"[chat] {args.prompt}"]
    raw_rewrite = post_edit_answers[f"[raw ] {args.prompt}"]
    chat_loc = post_edit_answers[f"[chat] {args.loc_prompt}"]
    # Official teacher-forcing rewrite_acc can stay 0 on this model (eval
    # forces left padding + exact first-token " Shanghai"). Contrast with
    # vanilla generate-match and free-generation substring checks.
    try:
        vanilla_rewrite = official_vanilla_acc(
            gen_model, tok, hparams, args.prompt, args.target, hparams.device
        )
    except Exception as exc:
        vanilla_rewrite = f"error: {exc}"
    vanilla_metrics = {
        "rewrite_acc_teacher_forcing": (m.get("post") or {}).get("rewrite_acc"),
        "rewrite_acc_vanilla_generation": vanilla_rewrite,
        "rewrite_contains_target_chat": contains_token_span(chat_rewrite, args.target),
        "rewrite_contains_target_raw": contains_token_span(raw_rewrite, args.target),
        "locality_contains_gt_chat": contains_token_span(chat_loc, args.loc_gt),
    }
    print("[metrics] ===== vanilla vs teacher-forcing =====")
    print(json.dumps(vanilla_metrics, indent=2, ensure_ascii=False, default=str))

    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    layer = None
    if hparams.inner_params:
        import re

        found = re.search(r"layers\[(\d+)\]", hparams.inner_params[0])
        if found:
            layer = int(found.group(1))
    report = {
        "model_path": hparams.model_name,
        "layer": layer,
        "act_ratio": hparams.act_ratio,
        "padding_side": hparams.padding_side,
        "enable_thinking": hparams.enable_thinking,
        "small_context": args.small_context,
        "case": {
            "prompt": args.prompt,
            "target_new": args.target,
            "ground_truth": args.ground_truth,
            "subject": args.subject,
        },
        "metrics": m,
        "vanilla_metrics": vanilla_metrics,
        "post_edit_answers": post_edit_answers,
        "edit_seconds": round(edit_seconds, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "peak_vram_GB": round(peak_gb, 1),
        "gpu_index": gpu_idx,
        "gpu_free_GB_before": round(gpu_free_gb, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"03_wise_edit{tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"[done] wrote {out_path} (edit {edit_seconds:.0f}s, peak {peak_gb:.1f}GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
