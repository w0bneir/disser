from __future__ import annotations

import unittest

import numpy as np

from anti_repetition import (
    VNFParameters,
    adaptive_schedule,
    assemble_sequence,
    loudness_match_sequences,
    no_repeat_schedule,
    pitch_gain_variant,
    technical_gate,
    velvet_spectral_variant,
)


class AntiRepetitionTests(unittest.TestCase):
    sample_rate = 44_100

    def make_hit(self) -> np.ndarray:
        rng = np.random.default_rng(17)
        time = np.arange(self.sample_rate, dtype=np.float64) / self.sample_rate
        audio = np.exp(-8.0 * time) * (
            0.7 * rng.standard_normal(time.size) + 0.3 * np.sin(2 * np.pi * 240 * time)
        )
        audio[0] = 1.0
        return (audio / np.max(np.abs(audio))).astype(np.float32)

    def test_velvet_variant_is_deterministic_safe_and_changed(self) -> None:
        reference = self.make_hit()
        parameters = VNFParameters(2.0, 8, 16.0, 0.09, 900.0, 42)
        first = velvet_spectral_variant(reference, self.sample_rate, parameters)
        second = velvet_spectral_variant(reference, self.sample_rate, parameters)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(reference, first))
        passed, failures = technical_gate(reference, first)
        self.assertTrue(passed, failures)

    def test_no_repeat_schedule_has_no_adjacent_duplicates(self) -> None:
        schedule = no_repeat_schedule(6, 20, 17)
        self.assertEqual(len(schedule), 20)
        self.assertTrue(all(left != right for left, right in zip(schedule, schedule[1:])))

    def test_adaptive_schedule_is_deterministic(self) -> None:
        reference = self.make_hit()
        candidates = [
            velvet_spectral_variant(
                reference,
                self.sample_rate,
                VNFParameters(1.5 + 0.1 * index, 7, 14.0, 0.07, 500 + 100 * index, index),
            )
            for index in range(8)
        ]
        first = adaptive_schedule(candidates, reference, self.sample_rate, count=20)
        second = adaptive_schedule(candidates, reference, self.sample_rate, count=20)
        self.assertEqual(first, second)
        self.assertTrue(all(left != right for left, right in zip(first, first[1:])))

    def test_sequences_share_length_rms_and_safe_peak(self) -> None:
        reference = self.make_hit()
        shifted = pitch_gain_variant(reference, semitones=0.5, gain_db=-0.4)
        sequences = {
            "repeat": assemble_sequence([reference] * 5, self.sample_rate, interval_ms=450.0),
            "changed": assemble_sequence([shifted] * 5, self.sample_rate, interval_ms=450.0),
        }
        matched = loudness_match_sequences(sequences, anchor_name="repeat")
        self.assertEqual(matched["repeat"].shape, matched["changed"].shape)
        rms_values = [float(np.sqrt(np.mean(np.square(values)))) for values in matched.values()]
        self.assertAlmostEqual(rms_values[0], rms_values[1], places=6)
        self.assertLessEqual(max(float(np.max(np.abs(v))) for v in matched.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
