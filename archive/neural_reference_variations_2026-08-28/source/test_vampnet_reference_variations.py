"""CPU-only проверки prompt-free VampNet-прототипа."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from vampnet_reference_variations import (
    MODEL_ASSET_SHA256,
    MODEL_ASSET_SIZES,
    build_fine_reconciliation_mask,
    build_reference_mask,
    build_codebook_hybrids,
    build_tiered_reference_mask,
    fix_length,
    match_rms_and_limit,
    prepare_codec_input,
    technical_audio_gate,
    validate_model_assets,
)
from run_vampnet_reference_variations import write_supervisor_demo


class VampNetReferenceVariationTests(unittest.TestCase):
    def test_conservative_mask_preserves_lower_codebooks_anchors_and_attack(self) -> None:
        mask = build_reference_mask(
            (1, 14, 21),
            upper_codebook_mask=3,
            periodic_prompt=7,
            periodic_offset=2,
            attack_tokens=2,
        )
        self.assertEqual(mask.shape, (1, 14, 21))
        self.assertTrue(np.all(mask[:, :3, :] == 0))
        self.assertTrue(np.all(mask[:, :, :2] == 0))
        self.assertTrue(np.all(mask[:, :, 2::7] == 0))
        self.assertEqual(mask[0, 3, 3], 1)

    def test_fix_length_trims_and_pads(self) -> None:
        source = np.arange(5, dtype=np.float32)
        np.testing.assert_array_equal(fix_length(source, 3), [0, 1, 2])
        np.testing.assert_array_equal(fix_length(source, 7), [0, 1, 2, 3, 4, 0, 0])

    def test_tiered_mask_moves_change_to_mid_and_sparsifies_fine(self) -> None:
        mask = build_tiered_reference_mask(
            (1, 14, 20),
            coarse_start=2,
            coarse_stop=4,
            coarse_anchor_period=5,
            coarse_anchor_offset=1,
            fine_start=4,
            fine_resample_period=4,
            fine_resample_offset=2,
            attack_tokens=2,
        )
        self.assertTrue(np.all(mask[:, :2, :] == 0))
        self.assertTrue(np.all(mask[:, :, :2] == 0))
        self.assertTrue(np.all(mask[:, 2:4, 1::5] == 0))
        self.assertEqual(mask[0, 2, 3], 1)
        self.assertTrue(np.all(mask[:, 4:, 2::4] == 1))
        self.assertEqual(mask[0, 4, 3], 0)

    def test_fine_reconciliation_mask_never_changes_event_codebooks_or_attack(self) -> None:
        mask = build_fine_reconciliation_mask(
            (1, 14, 12),
            fine_start=4,
            resample_period=3,
            resample_offset=1,
            attack_tokens=2,
        )
        self.assertTrue(np.all(mask[:, :4, :] == 0))
        self.assertTrue(np.all(mask[:, :, :2] == 0))
        self.assertTrue(np.all(mask[:, 4:, 4::3] == 1))
        self.assertTrue(np.all(mask[:, 4:, 3::3] == 0))

    def test_codebook_hybrids_replace_only_named_levels(self) -> None:
        reference = np.zeros((1, 14, 3), dtype=np.int64)
        variation = np.ones((1, 14, 3), dtype=np.int64)
        hybrids = build_codebook_hybrids(reference, variation)
        self.assertTrue(np.all(hybrids["codec_reference"] == 0))
        self.assertTrue(np.all(hybrids["cb1_only"][:, 1, :] == 1))
        self.assertTrue(np.all(hybrids["cb1_only"][:, 2:, :] == 0))
        self.assertTrue(np.all(hybrids["cb2_3_only"][:, 2:4, :] == 1))
        self.assertTrue(np.all(hybrids["cb2_3_only"][:, :2, :] == 0))
        self.assertTrue(np.all(hybrids["fine_4_13_only"][:, 4:, :] == 1))
        self.assertTrue(np.all(hybrids["full_variation"] == 1))

    def test_rms_matching_and_limiter(self) -> None:
        reference = np.asarray([0.25, -0.25] * 100, dtype=np.float32)
        candidate = np.asarray([0.01, -0.01] * 100, dtype=np.float32)
        matched = match_rms_and_limit(candidate, reference)
        self.assertAlmostEqual(float(np.sqrt(np.mean(matched**2))), 0.25, places=5)
        self.assertLessEqual(float(np.max(np.abs(matched))), 0.99)

    def test_codec_input_normalization_is_finite_and_bounded(self) -> None:
        source = np.asarray([0.2, -0.2] * 100, dtype=np.float32)
        prepared = prepare_codec_input(source)
        expected_rms = 10.0 ** (-24.0 / 20.0)
        self.assertAlmostEqual(float(np.sqrt(np.mean(prepared**2))), expected_rms, places=5)
        self.assertLessEqual(float(np.max(np.abs(prepared))), 0.99)

    def test_technical_gate_rejects_wrong_length_and_nan(self) -> None:
        reference = np.ones(10, dtype=np.float32) * 0.1
        candidate = np.ones(9, dtype=np.float32) * 0.1
        candidate[0] = np.nan
        passed, failures = technical_audio_gate(reference, candidate)
        self.assertFalse(passed)
        self.assertEqual(len(failures), 2)

    def test_asset_validation_checks_exact_size(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            codec = root / "codec.pth"
            expected_size = MODEL_ASSET_SIZES["codec.pth"]
            expected_hash = MODEL_ASSET_SHA256["codec.pth"]
            try:
                MODEL_ASSET_SIZES["codec.pth"] = 2
                MODEL_ASSET_SHA256["codec.pth"] = __import__("hashlib").sha256(b"xx").hexdigest()
                codec.write_bytes(b"x")
                with self.assertRaisesRegex(ValueError, "Неверный размер"):
                    validate_model_assets(root, required=("codec.pth",))
                codec.write_bytes(b"xx")
                report = validate_model_assets(root, required=("codec.pth",))
                self.assertEqual(report["codec.pth"]["bytes"], 2)
            finally:
                MODEL_ASSET_SIZES["codec.pth"] = expected_size
                MODEL_ASSET_SHA256["codec.pth"] = expected_hash

    def test_supervisor_demo_references_all_audio_files(self) -> None:
        report = {
            "variations": [
                {"file": "variation_01.wav", "seed": 17, "seconds": 0.5},
                {"file": "variation_02.wav", "seed": 42, "seconds": 0.4},
            ],
            "configuration": {"attack_ms": 80.0},
            "peak_vram_mib": 3682.0,
        }
        with TemporaryDirectory() as directory:
            path = write_supervisor_demo(Path(directory), report)
            page = path.read_text(encoding="utf-8")
            self.assertIn("reference_mono_44100.wav", page)
            self.assertIn("codec_roundtrip.wav", page)
            self.assertIn("variation_01.wav", page)
            self.assertIn("variation_02.wav", page)


if __name__ == "__main__":
    unittest.main()
