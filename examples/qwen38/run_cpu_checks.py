#!/usr/bin/env python
"""CPU-only checks you can run while other people occupy the GPUs.

Does not load 27B weights, does not set CUDA_VISIBLE_DEVICES.

    python examples/qwen38/run_cpu_checks.py
"""
from __future__ import annotations

import py_compile
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


PY_FILES = [
    ROOT / "easyeditor/util/model_loader.py",
    ROOT / "easyeditor/editors/editor.py",
    ROOT / "easyeditor/models/wise/WISE.py",
    ROOT / "easyeditor/models/wise/wise_hparams.py",
    EXAMPLES / "00_check_env.py",
    EXAMPLES / "01_smoke_load.py",
    EXAMPLES / "02_explore_modules.py",
    EXAMPLES / "03_wise_edit.py",
    EXAMPLES / "04_layer_scan.py",
    EXAMPLES / "05_ft_edit.py",
    EXAMPLES / "gpu_guard.py",
    EXAMPLES / "run_cpu_checks.py",
    ROOT / "tests/test_wise_label_mask.py",
    ROOT / "tests/test_qwen35_vl_loader.py",
]

YAML_LOADERS = [
    ("WISE", "easyeditor.models.wise.wise_hparams", "WISEHyperParams", ROOT / "hparams/WISE/qwen3.8-27b.yaml"),
    ("FT", "easyeditor.models.ft.ft_hparams", "FTHyperParams", ROOT / "hparams/FT/qwen3.8-27b.yaml"),
    ("GRACE", "easyeditor.models.grace.grace_hparams", "GraceHyperParams", ROOT / "hparams/GRACE/qwen3.8-27b.yaml"),
    ("WISE-qwen2.5", "easyeditor.models.wise.wise_hparams", "WISEHyperParams", ROOT / "hparams/WISE/qwen2.5-7b.yaml"),
]


def compile_files():
    for path in PY_FILES:
        if not path.is_file():
            print(f"[skip] missing {path.relative_to(ROOT)}")
            continue
        py_compile.compile(str(path), doraise=True)
        print(f"[ok  ] py_compile {path.relative_to(ROOT)}")


def load_yaml():
    import importlib

    for tag, module_name, cls_name, yaml_path in YAML_LOADERS:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        hp = cls.from_hparams(str(yaml_path))
        print(f"[ok  ] {tag} yaml → {cls_name} alg={hp.alg_name} model={hp.model_name}")


def run_unittests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in ("test_wise_label_mask.py", "test_qwen35_vl_loader.py"):
        suite.addTests(loader.discover(str(ROOT / "tests"), pattern=name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def git_diff_check():
    try:
        subprocess.check_call(["git", "diff", "--check"], cwd=str(ROOT))
    except subprocess.CalledProcessError:
        print("[warn] git diff --check found whitespace issues")
        return
    print("[ok  ] git diff --check")


def main() -> int:
    print("=== Qwen3.8 CPU checks (no GPU) ===")
    compile_files()
    load_yaml()
    run_unittests()
    git_diff_check()
    print("[done] CPU checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
