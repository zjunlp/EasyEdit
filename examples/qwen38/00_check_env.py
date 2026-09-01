#!/usr/bin/env python
"""Read-only environment check for Qwen3.8-27B EasyEdit runs.

Prints torch / transformers versions and per-GPU free memory (nvidia-smi).
Does not load weights and does not set CUDA_VISIBLE_DEVICES.

Requires transformers >= 5.8 (conda env EasyEdit-next).

Usage:
    conda activate EasyEdit-next
    python examples/qwen38/00_check_env.py
"""
from __future__ import annotations

import subprocess
import sys


MIN_TRANSFORMERS = (5, 8, 0)


def _version_tuple(version: str):
    parts = []
    for piece in version.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def query_gpus():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except FileNotFoundError:
        print("[gpu ] nvidia-smi not found")
        return []
    except subprocess.CalledProcessError as exc:
        print(f"[gpu ] nvidia-smi failed: {exc}")
        return []

    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, used, free = [x.strip() for x in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
                "free_gb": int(free) / 1024,
            }
        )
    return gpus


def main() -> int:
    print("[env ] python", sys.version.replace("\n", " "))
    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"[env ] missing import: {exc}")
        print("[env ] conda activate EasyEdit-next")
        return 1

    print(f"[env ] torch={torch.__version__}  cuda={torch.version.cuda}")
    print(f"[env ] transformers={transformers.__version__}")

    ver = _version_tuple(transformers.__version__)
    if ver < MIN_TRANSFORMERS:
        print(
            f"[env ] transformers {transformers.__version__} < 5.8; "
            "Qwen3.8-27B needs AutoModelForImageTextToText + language_model_only. "
            "conda activate EasyEdit-next (see requirements-qwen38.txt)"
        )
        return 1

    gpus = query_gpus()
    if not gpus:
        print("[gpu ] no GPUs reported")
    for gpu in gpus:
        print(
            f"[gpu ] GPU{gpu['index']} {gpu['name']}: "
            f"used {gpu['used_mib']} / {gpu['total_mib']} MiB, "
            f"free {gpu['free_gb']:.1f} GB"
        )
        if gpu["free_gb"] < 66.0:
            print(
                f"[gpu ] GPU{gpu['index']} free < 66GB — smoke/edit scripts will refuse this card"
            )

    print("[ok  ] environment looks usable for Qwen3.8 scripts (weights not loaded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
