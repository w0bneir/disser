"""CPU tests for late-tail spectral morphing."""

from __future__ import annotations

import unittest

import numpy as np

from spectral_tail_morph import SpectralTailMorphParameters, generate_spectral_tail_morph


class SpectralTailMorphTests(unittest.TestCase):
    def test_protected_energy_core_is_exact_and_tail_changes(self) -> None:
        sample_rate = 8_000
        time_axis = np.arange(8_000, dtype=np.float32) / sample_rate
        envelope = np.exp(-4 * time_axis)
        reference_mono = 0.6 * envelope * (
            np.sin(2 * np.pi * 220 * time_axis) + 0.3 * np.sin(2 * np.pi * 880 * time_axis)
        )
        generated_mono = 0.6 * envelope * (
            np.sin(2 * np.pi * 330 * time_axis) * (1 + 0.4 * np.sin(2 * np.pi * 3 * time_axis))
        )
        reference = np.stack((reference_mono, reference_mono), axis=1).astype(np.float32)
        generated = np.stack((generated_mono, generated_mono), axis=1).astype(np.float32)
        variation, diagnostics = generate_spectral_tail_morph(
            reference,
            generated,
            sample_rate,
            parameters=SpectralTailMorphParameters(
                protected_energy_quantile=0.75,
                transition_ms=40,
                n_fft=512,
                hop_length=128,
                frequency_smoothing_bins=5,
                time_smoothing_frames=5,
            ),
        )
        protected_frames = int(diagnostics["protected_frames"])
        self.assertEqual(variation.shape, reference.shape)
        np.testing.assert_array_equal(variation[:protected_frames], reference[:protected_frames])
        self.assertFalse(np.allclose(variation[protected_frames + 500 :], reference[protected_frames + 500 :]))
        self.assertEqual(diagnostics["core_max_abs_error"], 0.0)
        self.assertLessEqual(float(np.max(np.abs(variation))), 1.000001)

    def test_invalid_quantile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_spectral_tail_morph(
                np.ones((1_000, 1), dtype=np.float32) * 0.1,
                np.ones((1_000, 1), dtype=np.float32) * 0.1,
                8_000,
                parameters=SpectralTailMorphParameters(protected_energy_quantile=0.5),
            )

    def test_invalid_phase_mix_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_spectral_tail_morph(
                np.ones((1_000, 1), dtype=np.float32) * 0.1,
                np.ones((1_000, 1), dtype=np.float32) * 0.1,
                8_000,
                parameters=SpectralTailMorphParameters(phase_mix=1.1),
            )


if __name__ == "__main__":
    unittest.main()
