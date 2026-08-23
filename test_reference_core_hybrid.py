"""CPU tests for the reference-core hybrid method."""

from __future__ import annotations

import unittest

import numpy as np

from reference_core_hybrid import HybridParameters, generate_reference_core_hybrid


class ReferenceCoreHybridTests(unittest.TestCase):
    def test_core_is_exact_and_tail_changes(self) -> None:
        sample_rate = 8_000
        time_axis = np.arange(8_000, dtype=np.float32) / sample_rate
        reference_mono = np.sin(2 * np.pi * 220 * time_axis) * np.exp(-3 * time_axis)
        generated_mono = np.sin(2 * np.pi * 310 * time_axis + 0.4) * np.exp(-2 * time_axis)
        reference = np.stack((reference_mono, reference_mono), axis=1).astype(np.float32)
        generated = np.stack((generated_mono, -generated_mono), axis=1).astype(np.float32)
        parameters = HybridParameters(core_ms=100, transition_ms=50, residual_mix=0.2)

        variation, diagnostics = generate_reference_core_hybrid(
            reference,
            generated,
            sample_rate,
            parameters=parameters,
        )

        core_frames = round(0.1 * sample_rate)
        self.assertEqual(variation.shape, reference.shape)
        np.testing.assert_array_equal(variation[:core_frames], reference[:core_frames])
        self.assertFalse(np.allclose(variation[core_frames + 500 :], reference[core_frames + 500 :]))
        self.assertEqual(diagnostics["core_max_abs_error"], 0.0)
        self.assertLessEqual(float(np.max(np.abs(variation))), 1.000001)

    def test_mono_generated_can_feed_stereo_reference(self) -> None:
        reference = np.zeros((2_000, 2), dtype=np.float32)
        reference[0, :] = 0.9
        generated = np.full(2_000, 0.1, dtype=np.float32)
        variation, _ = generate_reference_core_hybrid(
            reference,
            generated,
            8_000,
            parameters=HybridParameters(core_ms=10, transition_ms=10),
        )
        self.assertEqual(variation.shape, reference.shape)
        self.assertTrue(np.isfinite(variation).all())

    def test_invalid_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_reference_core_hybrid(
                np.zeros((100, 1), dtype=np.float32),
                np.zeros((99, 1), dtype=np.float32),
                8_000,
                parameters=HybridParameters(core_ms=1),
            )


if __name__ == "__main__":
    unittest.main()
