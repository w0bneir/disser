"""CPU-only tests for the task-aligned LoRA mask."""

from __future__ import annotations

import unittest

import numpy as np

from train_vampnet_sfx_lora_pilot import build_task_aligned_mask


class VampNetSfxLoraPilotTests(unittest.TestCase):
    def test_mask_preserves_event_codebook_attack_and_individual_anchors(self) -> None:
        mask = build_task_aligned_mask(
            (2, 4, 20),
            attack_tokens=3,
            anchor_periods=np.asarray([5, 7]),
            anchor_offsets=np.asarray([1, 2]),
        )
        self.assertTrue(np.all(mask[:, 0, :] == 0))
        self.assertTrue(np.all(mask[:, :, :3] == 0))
        self.assertTrue(np.all(mask[0, 1:4, 6::5] == 0))
        self.assertTrue(np.all(mask[1, 1:4, 9::7] == 0))
        self.assertEqual(mask[0, 1, 4], 1)
        self.assertEqual(mask[1, 3, 4], 1)

    def test_mask_rejects_non_coarse_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "expects shape"):
            build_task_aligned_mask(
                (1, 14, 20),
                attack_tokens=3,
                anchor_periods=np.asarray([7]),
                anchor_offsets=np.asarray([0]),
            )


if __name__ == "__main__":
    unittest.main()
