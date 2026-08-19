"""CPU-проверки воспроизводимого DSP-baseline."""

from __future__ import annotations

import unittest

import numpy as np

from dsp_baseline import DspRanges, generate_dsp_variation, parameters_from_seed


class DspBaselineTests(unittest.TestCase):
    def test_parameters_are_reproducible_and_within_ranges(self) -> None:
        ranges = DspRanges()
        first = parameters_from_seed(42, ranges)
        second = parameters_from_seed(42, ranges)
        self.assertEqual(first, second)
        self.assertLessEqual(abs(first.pitch_cents), ranges.pitch_cents)
        self.assertLessEqual(
            abs(first.time_stretch_factor - 1),
            ranges.time_stretch_fraction,
        )
        self.assertLessEqual(abs(first.eq_gain_db), ranges.eq_gain_db)
        self.assertGreaterEqual(first.eq_center_hz, ranges.eq_center_min_hz)
        self.assertLessEqual(first.eq_center_hz, ranges.eq_center_max_hz)

    def test_variation_has_exact_shape_finite_values_and_matched_rms(self) -> None:
        sample_rate = 16_000
        time_axis = np.arange(8_000, dtype=np.float32) / sample_rate
        mono = (
            np.sin(2 * np.pi * 330 * time_axis)
            * np.exp(-2 * time_axis)
        ).astype(np.float32)
        stereo = np.stack((mono, mono * 0.8), axis=1)
        parameters = parameters_from_seed(17, DspRanges())
        variation = generate_dsp_variation(
            stereo,
            sample_rate,
            parameters=parameters,
        )
        self.assertEqual(variation.shape, stereo.shape)
        self.assertTrue(np.isfinite(variation).all())
        self.assertFalse(np.allclose(variation, stereo, atol=1e-4))
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(variation**2))),
            float(np.sqrt(np.mean(stereo**2))),
            places=4,
        )

    def test_invalid_ranges_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DspRanges(eq_center_max_hz=9_000).validate(16_000)


if __name__ == "__main__":
    unittest.main()

