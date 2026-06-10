#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class DummyHParams:
    alg_name = "IKE"
    device = 0


def fake_icl_lm_eval(model, model_name, hparams, tok, icl_examples, target, x, neighborhood=False):
    if neighborhood:
        return [target, x]
    return {
        "target": target,
        "prompt": x,
        "icl_examples": list(icl_examples),
    }


def build_record(portability_count):
    prompts = [f"Portability prompt {idx}" for idx in range(portability_count)]
    answers = [f"answer-{idx}" for idx in range(portability_count)]
    return {
        "prompt": "Who founded ExampleCorp?",
        "target_new": "Alice",
        "ground_truth": "Bob",
        "portability": {
            "por_hop": {
                "prompt": prompts,
                "ground_truth": answers,
            }
        },
    }


def run_case(evaluate_module, pre_edit, portability_count):
    metrics = evaluate_module.compute_icl_edit_quality(
        model=None,
        model_name="qwen2.5",
        hparams=DummyHParams(),
        tok=None,
        icl_examples=["Demo fact\n"],
        record=build_record(portability_count),
        device=0,
        pre_edit=pre_edit,
    )
    return metrics["portability"]["por_hop_acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portability-count", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    evaluate_module = importlib.import_module("easyeditor.evaluate.evaluate")
    original_icl_lm_eval = evaluate_module.icl_lm_eval
    evaluate_module.icl_lm_eval = fake_icl_lm_eval
    try:
        pre_acc = run_case(evaluate_module, pre_edit=True, portability_count=args.portability_count)
        post_acc = run_case(evaluate_module, pre_edit=False, portability_count=args.portability_count)
    finally:
        evaluate_module.icl_lm_eval = original_icl_lm_eval

    result = {
        "expected_count": args.portability_count,
        "pre_count": len(pre_acc),
        "post_count": len(post_acc),
        "pre_portability": pre_acc,
        "post_portability": post_acc,
    }
    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))

    if len(pre_acc) != args.portability_count or len(post_acc) != args.portability_count:
        raise SystemExit(
            "IKE portability list validation failed: "
            f"expected {args.portability_count}, got pre={len(pre_acc)}, post={len(post_acc)}"
        )


if __name__ == "__main__":
    main()
