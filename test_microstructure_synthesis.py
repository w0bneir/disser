import unittest

import numpy as np
from scipy import signal

from microstructure_synthesis import (
    PROTECT_UNTIL_S,
    calibrate_microstructure_strength,
    fit_microstructure_profile,
    leading_attack_error,
    microstructure_descriptor,
    microstructure_distance,
    synthesize_microstructure,
)


SR = 44_100


def _take(seed: int, *, colour: float = 0.92, decay: float = 3.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = int(round(2.5 * SR))
    time = np.arange(frames, dtype=np.float64) / SR
    onset = int(round(0.020 * SR))
    envelope = np.where(time >= 0.020, np.exp(-decay * (time - 0.020)), 0.0)
    excitation = signal.lfilter([1.0], [1.0, -colour], rng.standard_normal(frames))
    mono = 0.035 * excitation * envelope
    mono[onset : onset + 4] += np.asarray([0.8, -0.45, 0.20, -0.08])
    side = 0.05 * np.roll(mono, 9 + seed % 3)
    return np.stack((mono + side, mono - side), axis=1).astype(np.float32)


class MicrostructureSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.bank = [
            _take(1, colour=0.900, decay=3.1),
            _take(2, colour=0.910, decay=3.3),
            _take(3, colour=0.920, decay=3.5),
            _take(4, colour=0.930, decay=3.7),
            _take(5, colour=0.940, decay=3.9),
            _take(6, colour=0.950, decay=4.1),
        ]

    def test_descriptor_is_finite_and_excludes_attack(self):
        descriptor = microstructure_descriptor(self.bank[0], SR)
        self.assertEqual(descriptor.shape, (90,))
        self.assertTrue(np.isfinite(descriptor).all())
        changed_attack = self.bank[0].copy()
        changed_attack[int(0.020 * SR) : int(0.050 * SR)] *= -0.4
        np.testing.assert_allclose(
            microstructure_descriptor(changed_attack, SR),
            descriptor,
            rtol=0.0,
            atol=0.0,
        )

    def test_descriptor_does_not_collapse_antiphase_stereo(self):
        mono = self.bank[0][:, 0]
        antiphase = np.stack((mono, -mono), axis=1)
        descriptor = microstructure_descriptor(antiphase, SR)
        self.assertTrue(np.isfinite(descriptor).all())
        # A destructive L+R fold would make all spectral-band values equal.
        self.assertGreater(float(np.std(descriptor[:8])), 0.05)

    def test_profile_uses_only_within_group_pairs(self):
        profile = fit_microstructure_profile(
            self.bank,
            SR,
            groups=["a", "a", "a", "b", "b", "b"],
            names=[f"take_{index}" for index in range(6)],
        )
        self.assertEqual(profile.pairwise_distances.size, 6)
        self.assertEqual(set(profile.pairwise_groups), {"a", "b"})
        self.assertGreater(profile.corridor_high, profile.corridor_low)
        self.assertTrue(np.isfinite(profile.descriptor_scale).all())

    def test_new_seed_changes_body_but_never_attack(self):
        first = synthesize_microstructure(self.bank[0], SR, seed=17, strength=0.45)
        second = synthesize_microstructure(self.bank[0], SR, seed=42, strength=0.45)
        self.assertEqual(first.shape, self.bank[0].shape)
        self.assertEqual(leading_attack_error(self.bank[0], first, SR), 0.0)
        protected = int(round(PROTECT_UNTIL_S * SR))
        np.testing.assert_array_equal(first[:protected], second[:protected])
        self.assertGreater(float(np.linalg.norm(first[protected:] - second[protected:])), 0.01)
        self.assertTrue(np.isfinite(first).all())

    def test_strength_is_monotonic_in_waveform_difference(self):
        low = synthesize_microstructure(self.bank[0], SR, seed=17, strength=0.15)
        high = synthesize_microstructure(self.bank[0], SR, seed=17, strength=0.45)
        low_delta = np.linalg.norm(low - self.bank[0])
        high_delta = np.linalg.norm(high - self.bank[0])
        self.assertGreater(high_delta, low_delta * 2.5)

    def test_microstructure_distance_is_positive(self):
        profile = fit_microstructure_profile(
            self.bank,
            SR,
            groups=["a", "a", "a", "b", "b", "b"],
        )
        candidate = synthesize_microstructure(self.bank[0], SR, seed=7, strength=0.35)
        distance = microstructure_distance(self.bank[0], candidate, profile)
        self.assertTrue(np.isfinite(distance))
        self.assertGreater(distance, 0.0)

    def test_strength_calibration_approaches_natural_target(self):
        profile = fit_microstructure_profile(
            self.bank,
            SR,
            groups=["a", "a", "a", "b", "b", "b"],
        )
        candidate, strength, distance = calibrate_microstructure_strength(
            self.bank[0],
            profile,
            seed=17,
            target_distance=profile.corridor_low,
            iterations=7,
        )
        self.assertGreater(strength, 0.0)
        self.assertLessEqual(strength, 0.75)
        self.assertEqual(candidate.shape, self.bank[0].shape)
        self.assertLess(abs(distance - profile.corridor_low), profile.corridor_low * 0.20)

    def test_validation_and_zero_strength(self):
        zero = synthesize_microstructure(self.bank[0], SR, seed=1, strength=0.0)
        np.testing.assert_array_equal(zero, self.bank[0])
        with self.assertRaises(ValueError):
            synthesize_microstructure(self.bank[0], SR, seed=1, strength=1.01)
        with self.assertRaises(ValueError):
            fit_microstructure_profile(self.bank[:5], SR, groups=["a"] * 5)
        with self.assertRaises(ValueError):
            fit_microstructure_profile(self.bank, SR, groups=["a"] * 5)


if __name__ == "__main__":
    unittest.main()
