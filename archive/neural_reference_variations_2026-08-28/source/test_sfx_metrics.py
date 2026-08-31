"""CPU-проверки метрик reference-guided SFX-вариаций."""

from __future__ import annotations

import unittest

import torch

from sfx_metrics import (
    compare_to_reference,
    envelope_peak_times,
    pairwise_diversity,
    peak_timing_distance_seconds,
    spectral_features,
)


class SfxMetricsTests(unittest.TestCase):
    def test_identical_audio_is_detected_as_repeat(self) -> None:
        generator = torch.Generator().manual_seed(7)
        audio = torch.randn(16_000, generator=generator) * 0.1
        metrics = compare_to_reference(audio, audio.clone(), 16_000)
        self.assertAlmostEqual(float(metrics["envelope_pearson"]), 1.0, places=5)
        self.assertAlmostEqual(float(metrics["envelope_mse"]), 0.0, places=7)
        self.assertAlmostEqual(float(metrics["waveform_pearson"]), 1.0, places=5)
        self.assertLessEqual(float(metrics["copy_residual_db"]), -100)
        self.assertAlmostEqual(float(metrics["log_spectral_distance_db"]), 0.0, places=5)

    def test_peak_timing_distance_detects_shift(self) -> None:
        reference_envelope = torch.zeros(100)
        candidate_envelope = torch.zeros(100)
        reference_envelope[10] = 1
        reference_envelope[70] = 0.8
        candidate_envelope[20] = 1
        candidate_envelope[80] = 0.8
        reference_peaks = envelope_peak_times(
            reference_envelope,
            sample_rate=1_000,
            hop_length=10,
            minimum_distance_seconds=0.05,
        )
        candidate_peaks = envelope_peak_times(
            candidate_envelope,
            sample_rate=1_000,
            hop_length=10,
            minimum_distance_seconds=0.05,
        )
        distance = peak_timing_distance_seconds(
            reference_peaks,
            candidate_peaks,
            duration=1.0,
        )
        self.assertAlmostEqual(distance, 0.1, places=5)

    def test_spectral_centroid_tracks_frequency(self) -> None:
        sample_rate = 16_000
        time_axis = torch.arange(sample_rate) / sample_rate
        low = torch.sin(2 * torch.pi * 300 * time_axis)
        high = torch.sin(2 * torch.pi * 3_000 * time_axis)
        self.assertLess(
            spectral_features(low, sample_rate)["spectral_centroid_hz"],
            spectral_features(high, sample_rate)["spectral_centroid_hz"],
        )

    def test_repeat_batch_has_no_pairwise_diversity(self) -> None:
        audio = torch.linspace(-1, 1, 8_000)
        rows = pairwise_diversity([(17, audio), (42, audio.clone())], 16_000)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["waveform_pearson"]), 1.0, places=5)
        self.assertAlmostEqual(float(rows[0]["envelope_pearson"]), 1.0, places=5)
        self.assertAlmostEqual(float(rows[0]["log_spectral_distance_db"]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

