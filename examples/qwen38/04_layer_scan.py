#!/usr/bin/env python
"""Layer scan scaffold for Qwen3.8-27B WISE.

Loads once via BaseEditor + yaml, then edits a few layers independently
(restore after each edit). Distinguishes GDN / linear_attention (e.g. 53)
from full_attention (51/55/59/63). Layer 29 is known to fail free-gen
transfer (teacher-forcing can move, generation does not).

This script loads 27B weights. It picks the emptiest GPU and refuses to
run below 66GB free. Do not run it while other jobs occupy the cards.

Usage:
    python examples/qwen38/04_layer_scan.py \\
        --hparams hparams/WISE/qwen3.8-27b.yaml \\
        --model ./hugging_cache/Qwen3.8-27B
    python examples/qwen38/04_layer_scan.py --layers 51,53,55,59,63 --tag attn

Results:
    examples/qwen38/results/04_layer_scan[_<tag>].json

ROME is not claimed to work; see hparams/ROME/qwen3.8-27b.yaml.example.
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

# Mixed-attention map from config.text_config.layer_types (64 layers).
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
# Default scan: GDN 45/49/53/57/61 + full_attention 51/55/59/63.
DEFAULT_LAYERS = (45, 49, 51, 53, 55, 57, 59, 61, 63)
# Known negative: GDN layer 29 (~45% depth) does not transfer to free gen.
KNOWN_BAD_FREE_GEN_LAYERS = (29,)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument(
        "--model",
        default=None,
        help="Override yaml model_name (e.g. ./hugging_cache/Qwen3.8-27B)",
    )
    parser.add_argument(
        "--layers",
        default=",".join(str(x) for x in DEFAULT_LAYERS),
        help="Comma-separated layer indices to scan",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--target", default="Shanghai")
    parser.add_argument("--ground-truth", default="Paris")
    parser.add_argument("--subject", default="France")
    parser.add_argument("--loc-prompt", default="The capital of Japan is")
    parser.add_argument("--loc-gt", default="Tokyo")
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    return parser.parse_args()


def layer_kind(layer: int) -> str:
    if layer in FULL_ATTENTION_LAYERS:
        return "full_attention"
    return "linear_attention"


def parse_layers(spec: str):
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    if not layers:
        raise ValueError("no layers parsed from --layers")
    return layers


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


def main() -> int:
    args = parse_args()
    layers = parse_layers(args.layers)
    gpu_idx, gpu_free_gb = pick_gpu(args.min_free_gb)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch  # noqa: E402
    from easyeditor import BaseEditor  # noqa: E402
    from easyeditor.models.wise.wise_hparams import WISEHyperParams  # noqa: E402

    print(
        f"[gpu ] selected GPU{gpu_idx} (free {gpu_free_gb:.1f}GB); "
        f"CUDA_VISIBLE_DEVICES={gpu_idx}"
    )
    print(f"[scan] layers={layers}")
    for layer in layers:
        note = ""
        if layer in KNOWN_BAD_FREE_GEN_LAYERS:
            note = "  # known: teacher-forcing can move, free-gen does not"
        print(f"       L{layer} {layer_kind(layer)}{note}")

    hparams = WISEHyperParams.from_hparams(args.hparams)
    if args.model:
        hparams.model_name = args.model
    hparams.device = 0
    hparams.inner_params = [
        f"model.language_model.layers[{layers[0]}].mlp.down_proj.weight"
    ]

    t_load = time.time()
    editor = BaseEditor.from_hparams(hparams)
    print(f"[load] BaseEditor ready ({time.time() - t_load:.0f}s)")

    if hasattr(editor.model, "gradient_checkpointing_enable"):
        editor.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(editor.model, "config"):
            editor.model.config.use_cache = False

    tok = editor.tok
    per_layer = []
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    for layer in layers:
        kind = layer_kind(layer)
        editor.hparams.inner_params = [
            f"model.language_model.layers[{layer}].mlp.down_proj.weight"
        ]
        print(f"\n[edit] layer={layer} kind={kind}")
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
            sequential_edit=False,  # restore_after_edit between layers
        )
        edit_s = time.time() - t_edit
        m = metrics[0] if metrics else {}
        gen_model = unwrap_generate_model(edited_model)
        try:
            for mod in (editor.model, gen_model):
                if hasattr(mod, "gradient_checkpointing_disable"):
                    mod.gradient_checkpointing_disable()
                if hasattr(mod, "config"):
                    mod.config.use_cache = True
        except Exception as exc:
            print(f"[gen ] restore use_cache failed (ignored): {exc}")

        answers = {
            f"[chat] {args.prompt}": ask(tok, gen_model, args.prompt),
            f"[raw ] {args.prompt}": ask(tok, gen_model, args.prompt, chat=False),
            f"[chat] {args.loc_prompt}": ask(tok, gen_model, args.loc_prompt),
            f"[raw ] {args.loc_prompt}": ask(tok, gen_model, args.loc_prompt, chat=False),
        }
        print(json.dumps(answers, indent=2, ensure_ascii=False))
        per_layer.append(
            {
                "layer": layer,
                "kind": kind,
                "known_bad_free_gen": layer in KNOWN_BAD_FREE_GEN_LAYERS,
                "edit_seconds": round(edit_s, 1),
                "metrics": m,
                "post_edit_answers": answers,
            }
        )
        if hasattr(editor.model, "gradient_checkpointing_enable"):
            editor.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(editor.model, "config"):
                editor.model.config.use_cache = False

    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    report = {
        "model_path": hparams.model_name,
        "layers": layers,
        "full_attention_layers": list(FULL_ATTENTION_LAYERS),
        "known_bad_free_gen_layers": list(KNOWN_BAD_FREE_GEN_LAYERS),
        "rome_status": "scaffold only; see hparams/ROME/qwen3.8-27b.yaml.example",
        "per_layer": per_layer,
        "total_seconds": round(time.time() - t_start, 1),
        "peak_vram_GB": round(peak_gb, 1),
        "gpu_index": gpu_idx,
        "gpu_free_GB_before": round(gpu_free_gb, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"04_layer_scan{tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"[done] wrote {out_path} (peak {peak_gb:.1f}GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
