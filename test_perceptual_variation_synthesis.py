import unittest

import numpy as np
from scipy import signal

from perceptual_variation_synthesis import (
    common_peak_safe,
    estimate_transform,
    event_descriptor,
    fit_natural_variation_profile,
    profile_distance,
    synthesize_variation,
    waveform_correlation,
)


SR = 44_100


def _take(seed: int, *, tilt: float = 0.0, decay: float = 4.0, side: float = 0.12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = int(2.5 * SR)
    time = np.arange(frames) / SR
    onset = int(0.02 * SR)
    excitation = np.zeros(frames)
    excitation[onset] = 1.0
    noise = rng.standard_normal(frames)
    envelope = np.where(time >= 0.02, np.exp(-decay * (time - 0.02)), 0.0)
    body = signal.lfilter([1.0], [1.0, -0.94 + tilt], noise) * envelope * 0.025
    mono = excitation * 0.75 + body
    delayed = np.roll(mono, 7 + seed % 3)
    left = mono + side * delayed
    right = mono - side * delayed
    return np.stack((left, right), axis=1).astype(np.float32)


class PerceptualVariationSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.bank = [
            _take(1, tilt=-0.010, decay=3.7, side=0.10),
            _take(2, tilt=0.000, decay=4.0, side=0.12),
            _take(3, tilt=0.008, decay=4.3, side=0.14),
            _take(4, tilt=0.014, decay=4.6, side=0.16),
        ]

    def test_profile_is_finite_and_selects_a_medoid(self):
        profile = fit_natural_variation_profile(self.bank, SR, names=["a", "b", "c", "d"])
        self.assertEqual(profile.descriptors.shape[0], 4)
        self.assertTrue(np.isfinite(profile.descriptors).all())
        self.assertGreaterEqual(profile.reference_index, 0)
        self.assertLess(profile.reference_index, 4)
        self.assertGreater(profile.corridor_high, 0.0)
        self.assertGreaterEqual(profile.corridor_high, profile.corridor_low)

    def test_zero_strength_is_sample_identical(self):
        output, _ = synthesize_variation(self.bank[0], self.bank[3], SR, strength=0.0)
        np.testing.assert_array_equal(output, self.bank[0])

    def test_bounded_transform_changes_body_but_preserves_event(self):
        weak, transform = synthesize_variation(self.bank[0], self.bank[3], SR, strength=0.35)
        strong, _ = synthesize_variation(self.bank[0], self.bank[3], SR, strength=0.85)
        self.assertEqual(weak.shape, self.bank[0].shape)
        self.assertTrue(np.isfinite(strong).all())
        self.assertLessEqual(float(np.max(np.abs(transform.spectral_gain_db))), 3.5 + 1e-8)
        self.assertGreater(waveform_correlation(self.bank[0], weak), 0.95)
        attack_end = int(0.035 * SR)
        attack_error = np.sqrt(np.mean(np.square(strong[:attack_end] - self.bank[0][:attack_end])))
        body_error = np.sqrt(np.mean(np.square(strong[attack_end:] - self.bank[0][attack_end:])))
        self.assertLess(attack_error, body_error)
        self.assertGreater(np.linalg.norm(strong - self.bank[0]), np.linalg.norm(weak - self.bank[0]))

    def test_descriptor_distance_and_common_peak_safety(self):
        profile = fit_natural_variation_profile(self.bank, SR)
        output, _ = synthesize_variation(self.bank[profile.reference_index], self.bank[-1], SR, strength=0.7)
        distance = profile_distance(self.bank[profile.reference_index], output, profile)
        self.assertTrue(np.isfinite(distance))
        self.assertGreater(distance, 0.0)
        safe = common_peak_safe([self.bank[0] * 2.0, output * 2.0])
        self.assertLessEqual(max(float(np.max(np.abs(audio))) for audio in safe), 10 ** (-1 / 20) + 1e-6)

    def test_rejects_incompatible_inputs(self):
        with self.assertRaises(ValueError):
            fit_natural_variation_profile(self.bank[:2], SR)
        with self.assertRaises(ValueError):
            estimate_transform(self.bank[0], self.bank[1][:-1], SR)
        with self.assertRaises(ValueError):
            synthesize_variation(self.bank[0], self.bank[1], SR, strength=1.1)

    def test_descriptor_dimension_is_stable(self):
        self.assertEqual(event_descriptor(self.bank[0], SR).shape, (45,))


if __name__ == "__main__":
    unittest.main()
