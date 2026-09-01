#!/usr/bin/env python
"""CPU unit tests for WISE prompt/target location under left and right padding."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from easyeditor.models.wise.WISE import prompt_token_loc_from_labels


class PromptTokenLocTests(unittest.TestCase):
    def test_left_pad_matches_legacy_minus100_count(self):
        # pads + masked prompt, then target tokens
        labels = torch.tensor(
            [
                [-100, -100, -100, 11, 12, 13],
                [-100, -100, 21, 22, 23, 24],
            ]
        )
        loc = prompt_token_loc_from_labels(labels)
        legacy = (labels == -100).sum(dim=-1) - 1
        self.assertEqual(loc.tolist(), [2, 1])
        self.assertEqual(loc.tolist(), legacy.tolist())

    def test_right_pad_ignores_trailing_pads(self):
        # masked prompt, target, then trailing pads (right padding)
        labels = torch.tensor(
            [
                [-100, -100, 11, 12, -100, -100],
                [-100, 21, 22, 23, -100, -100],
            ]
        )
        loc = prompt_token_loc_from_labels(labels)
        legacy = (labels == -100).sum(dim=-1) - 1
        # first non-100 minus 1 == last prompt token
        self.assertEqual(loc.tolist(), [1, 0])
        # legacy counts trailing pads and is wrong on the right
        self.assertEqual(legacy.tolist(), [3, 2])
        self.assertNotEqual(loc.tolist(), legacy.tolist())

    def test_mixed_batch_left_and_right(self):
        labels = torch.tensor(
            [
                [-100, -100, -100, 7, 8, 9],      # left
                [-100, -100, 7, 8, 9, -100],      # right
            ]
        )
        loc = prompt_token_loc_from_labels(labels)
        self.assertEqual(loc.tolist(), [2, 1])

    def test_all_masked_row_uses_last_index(self):
        labels = torch.tensor([[-100, -100, -100, -100]])
        loc = prompt_token_loc_from_labels(labels)
        self.assertEqual(loc.tolist(), [3])


if __name__ == "__main__":
    unittest.main()
