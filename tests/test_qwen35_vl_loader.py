#!/usr/bin/env python
"""CPU tests for Qwen3.5 VL-text detection (no Hub, no GPU, no 27B weights)."""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from easyeditor.util.model_loader import (
    _config_is_qwen35_vl_text,
    _has_qwen38_path_hint,
    is_qwen35_vl_text_model,
)


def _cfg(**kwargs):
    return types.SimpleNamespace(**kwargs)


class PathHintTests(unittest.TestCase):
    def test_qwen38_dot_name(self):
        self.assertTrue(_has_qwen38_path_hint("./hugging_cache/Qwen3.8-27B"))
        self.assertTrue(_has_qwen38_path_hint("/abs/path/Qwen3.8-27B"))

    def test_qwen3_8b_not_matched(self):
        # Previous hint used "qwen3-8" and would false-positive Qwen3-8B.
        self.assertFalse(_has_qwen38_path_hint("./hugging_cache/Qwen3-8B"))
        self.assertFalse(_has_qwen38_path_hint("Qwen3-8B-Instruct"))
        self.assertFalse(_has_qwen38_path_hint("qwen3-8b"))

    def test_qwen35_9b_not_matched(self):
        self.assertFalse(_has_qwen38_path_hint("./hugging_cache/Qwen3.5-9B"))
        self.assertFalse(_has_qwen38_path_hint("Qwen2.5-7B-Instruct"))


class ConfigDetectTests(unittest.TestCase):
    def test_conditional_generation_arch(self):
        cfg = _cfg(
            architectures=["Qwen3_5ForConditionalGeneration"],
            model_type="qwen3_5",
            vision_config=object(),
            text_config=object(),
        )
        self.assertTrue(_config_is_qwen35_vl_text(cfg))

    def test_pure_text_qwen35_false(self):
        cfg = _cfg(
            architectures=["Qwen3_5ForCausalLM"],
            model_type="qwen3_5",
            vision_config=None,
            text_config=None,
        )
        self.assertFalse(_config_is_qwen35_vl_text(cfg))

    def test_qwen2_false(self):
        cfg = _cfg(architectures=["Qwen2ForCausalLM"], model_type="qwen2")
        self.assertFalse(_config_is_qwen35_vl_text(cfg))


class PublicDetectTests(unittest.TestCase):
    def test_non_qwen_skipped(self):
        self.assertFalse(is_qwen35_vl_text_model("llama-7b"))
        self.assertFalse(is_qwen35_vl_text_model(None))

    def test_missing_local_path_uses_hint_only(self):
        self.assertTrue(is_qwen35_vl_text_model("./hugging_cache/Qwen3.8-27B"))
        self.assertFalse(is_qwen35_vl_text_model("./hugging_cache/Qwen3-8B"))
        self.assertFalse(is_qwen35_vl_text_model("./hugging_cache/Qwen2.5-7b-Instruct"))

    def test_real_checkpoint_if_present(self):
        local = Path("./hugging_cache/Qwen3.8-27B")
        if not (local / "config.json").is_file():
            self.skipTest("local Qwen3.8-27B checkpoint not on this machine")
        self.assertTrue(is_qwen35_vl_text_model(str(local)))


if __name__ == "__main__":
    unittest.main()
