"""CPU-only tests for the active structured-resynthesis method."""

from __future__ import annotations

import unittest

import numpy as np

from structured_resynthesis import (
    analyze_event_regions,
    build_component_masks,
    cascade_allpass,
    diagnostic_metrics,
    generate_causal_variations,
    region_mask,
    technical_gate,
)


class StructuredResynthesisTests(unittest.TestCase):
    sample_rate = 44_100

    def make_one_shot(self) -> np.ndarray:
        rng = np.random.default_rng(17)
        time = np.arange(self.sample_rate, dtype=np.float64) / self.sample_rate
        decay = np.exp(-7.0 * time)
        carrier = 0.65 * rng.standard_normal(time.size) + 0.35 * np.sin(2 * np.pi * 310 * time)
        audio = decay * carrier
        audio[0] = 1.0
        return (audio / np.max(np.abs(audio))).astype(np.float32)

    def test_region_analysis_is_ordered_and_masks_are_bounded(self) -> None:
        audio = self.make_one_shot()
        regions = analyze_event_regions(audio, self.sample_rate)
        self.assertLess(regions.peak_sample, regions.attack_end_sample)
        self.assertLess(regions.attack_end_sample, regions.body_end_sample)
        self.assertLess(regions.body_end_sample, regions.frames)
        masks = build_component_masks(regions)
        self.assertEqual(set(masks), {"attack", "body", "tail"})
        for mask in masks.values():
            self.assertEqual(mask.shape, audio.shape)
            self.assertGreaterEqual(float(mask.min()), 0.0)
            self.assertLessEqual(float(mask.max()), 1.0)
        self.assertEqual(float(masks["attack"][0]), 1.0)
        self.assertEqual(float(masks["tail"][-1]), 1.0)

    def test_region_mask_rejects_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "границы"):
            region_mask(100, start=50, end=50, fade_in=2, fade_out=2)

    def test_allpass_is_finite_and_does_not_collapse_energy(self) -> None:
        audio = self.make_one_shot()
        output = cascade_allpass(
            audio,
            delays_samples=(17, 31),
            coefficients=(0.25, -0.2),
        )
        self.assertTrue(np.isfinite(output).all())
        ratio = np.sqrt(np.mean(output**2)) / np.sqrt(np.mean(audio**2))
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.2)

    def test_six_variations_are_deterministic_valid_and_nonidentical(self) -> None:
        audio = self.make_one_shot()
        regions = analyze_event_regions(audio, self.sample_rate)
        first = generate_causal_variations(audio, self.sample_rate, regions)
        second = generate_causal_variations(audio, self.sample_rate, regions)
        self.assertEqual(len(first), 6)
        self.assertEqual(list(first), list(second))
        for name, candidate in first.items():
            np.testing.assert_array_equal(candidate, second[name])
            passed, failures = technical_gate(audio, candidate)
            self.assertTrue(passed, failures)
            self.assertFalse(np.array_equal(audio, candidate))
            metrics = diagnostic_metrics(audio, candidate, self.sample_rate, regions)
            self.assertGreater(metrics["envelope_correlation"], 0.9)


if __name__ == "__main__":
    unittest.main()
