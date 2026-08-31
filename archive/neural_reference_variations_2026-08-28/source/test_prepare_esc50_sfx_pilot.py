"""CPU-only tests for deterministic ESC-50 event cropping."""

from __future__ import annotations

import unittest

import numpy as np

from prepare_esc50_sfx_pilot import SAMPLE_RATE, dominant_event_crop


class Esc50PilotPreparationTests(unittest.TestCase):
    def test_dominant_event_is_placed_after_fixed_preroll(self) -> None:
        source = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        impulse_sample = 2 * SAMPLE_RATE
        source[impulse_sample : impulse_sample + 512] = 0.75
        crop, diagnostics = dominant_event_crop(source, SAMPLE_RATE)
        self.assertEqual(crop.size, int(round(1.75 * SAMPLE_RATE)))
        crop_peak = int(np.argmax(np.abs(crop)))
        # The RMS frame centres the crop a few ms around the physical impulse.
        self.assertLess(abs(crop_peak / SAMPLE_RATE - 0.08), 0.02)
        self.assertGreater(float(diagnostics["peak"]), 0.7)

    def test_stereo_is_folded_to_mono_and_output_is_finite(self) -> None:
        mono = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mono[10_000:11_000] = 0.5
        stereo = np.stack([mono, -0.5 * mono], axis=1)
        crop, diagnostics = dominant_event_crop(stereo, SAMPLE_RATE)
        self.assertEqual(crop.ndim, 1)
        self.assertTrue(np.isfinite(crop).all())
        self.assertGreater(float(diagnostics["rms"]), 0.0)


if __name__ == "__main__":
    unittest.main()
