from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from sfx_pool_optimizer import (
    analyze_directory,
    analyze_file,
    assemble_sequence,
    build_distance_matrices,
    build_groupwise_distance_matrices,
    detect_onset,
    discover_wav_files,
    group_index_map,
    infer_group,
    rms_match_sequences,
    medoid_index,
    normalize_prepared_bank,
    perceptual_schedule,
    recommend_groups,
    schedule_diagnostics,
    select_representative_pool,
    shuffle_schedule,
    technical_audio_gate,
)


class SfxPoolOptimizerTests(unittest.TestCase):
    sample_rate = 8_000

    def make_shot(
        self,
        *,
        frequency: float = 700.0,
        decay: float = 7.0,
        leading_s: float = 0.05,
        side: float = 0.05,
        seed: int = 17,
    ) -> np.ndarray:
        duration_s = 1.4
        frames = int(round(duration_s * self.sample_rate))
        onset = int(round(leading_s * self.sample_rate))
        active_frames = frames - onset
        time = np.arange(active_frames, dtype=np.float64) / self.sample_rate
        rng = np.random.default_rng(seed)
        carrier = 0.55 * rng.standard_normal(active_frames) + 0.45 * np.sin(
            2.0 * np.pi * frequency * time
        )
        event = np.exp(-decay * time) * carrier
        event[0] += 2.5
        event /= max(float(np.max(np.abs(event))), 1e-9)
        output = np.zeros((frames, 2), dtype=np.float64)
        output[onset:, 0] = event
        output[onset:, 1] = (1.0 - side) * event + side * np.roll(event, 3)
        return (0.8 * output).astype(np.float32)

    def write_group(self, directory: Path, group: int, count: int) -> list[Path]:
        paths = []
        for index in range(1, count + 1):
            path = directory / f"SHOT {group}.{index}.wav"
            sf.write(
                path,
                self.make_shot(
                    frequency=500.0 + 80.0 * index + 20.0 * group,
                    decay=5.0 + 0.6 * index,
                    side=0.02 * index,
                    seed=group * 100 + index,
                ),
                self.sample_rate,
                subtype="PCM_24",
            )
            paths.append(path)
        return paths

    def test_discovery_and_group_parsing_are_natural_and_unicode_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_group(directory, 2, 3)
            paths = discover_wav_files(directory)
            self.assertEqual([path.name for path in paths], ["SHOT 2.1.wav", "SHOT 2.2.wav", "SHOT 2.3.wav"])
            self.assertEqual(infer_group(paths[0]), "2")

    def test_onset_detection_tolerates_leading_silence(self) -> None:
        audio = self.make_shot(leading_s=0.073)
        onset = detect_onset(audio, self.sample_rate) / self.sample_rate
        self.assertLess(abs(onset - 0.073), 0.008)

    def test_onset_and_features_survive_antiphase_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "SHOT 1.1.wav"
            audio = self.make_shot(leading_s=0.073)
            audio[:, 1] = -audio[:, 0]
            sf.write(path, audio, self.sample_rate, subtype="PCM_24")
            onset = detect_onset(audio, self.sample_rate) / self.sample_rate
            analyzed = analyze_file(path, clip_duration_s=1.2)
            self.assertLess(abs(onset - 0.073), 0.008)
            self.assertGreater(analyzed.metrics.early_rms_dbfs, -80.0)
            for features in analyzed.features.values():
                self.assertTrue(np.isfinite(features).all())

    def test_feature_extraction_is_finite_and_gain_robust_except_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_path = directory / "SHOT 1.1.wav"
            second_path = directory / "SHOT 1.2.wav"
            audio = self.make_shot()
            sf.write(first_path, audio, self.sample_rate, subtype="PCM_24")
            sf.write(second_path, 0.5 * audio, self.sample_rate, subtype="PCM_24")
            first = analyze_file(first_path, clip_duration_s=1.2)
            second = analyze_file(second_path, clip_duration_s=1.2)
            for features in first.features.values():
                self.assertTrue(np.isfinite(features).all())
            np.testing.assert_allclose(first.features["timbre"], second.features["timbre"], atol=2e-3)
            np.testing.assert_allclose(first.features["envelope"], second.features["envelope"], atol=2e-3)
            self.assertGreater(
                abs(first.features["level"][0] - second.features["level"][0]),
                5.5,
            )

    def test_distance_matrix_and_recommendation_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_group(directory, 1, 5)
            self.write_group(directory, 2, 3)
            clips = analyze_directory(directory, clip_duration_s=1.2)
            distances, components = build_distance_matrices(clips)
            self.assertEqual(distances.shape, (8, 8))
            np.testing.assert_allclose(distances, distances.T, atol=1e-12)
            np.testing.assert_array_equal(np.diag(distances), np.zeros(8))
            self.assertTrue(np.isfinite(distances).all())
            self.assertEqual(set(components), {"attack", "timbre", "envelope", "spatial", "level"})
            groups = group_index_map(clips)
            selected = select_representative_pool(groups["1"], distances, size=3)
            self.assertEqual(len(selected), 3)
            self.assertIn(medoid_index(groups["1"], distances), selected)
            self.assertEqual(select_representative_pool(groups["2"], distances, size=3), groups["2"])
            recommendations = recommend_groups(clips, distances, pool_size=3)
            self.assertEqual({item.group for item in recommendations}, {"1", "2"})

    def test_large_pool_selection_uses_bounded_greedy_fallback(self) -> None:
        positions = np.linspace(0.0, 1.0, 50, dtype=np.float64)
        distances = np.abs(positions[:, None] - positions[None, :])
        selected = select_representative_pool(range(50), distances, size=6)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(set(selected)), 6)
        self.assertIn(medoid_index(range(50), distances), selected)

    def test_groupwise_distance_is_invariant_to_unrelated_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_group(directory, 1, 5)
            first_clips = analyze_directory(directory, clip_duration_s=1.2)
            first_distances, _ = build_groupwise_distance_matrices(first_clips)
            self.write_group(directory, 9, 3)
            combined_clips = analyze_directory(directory, clip_duration_s=1.2)
            combined_distances, _ = build_groupwise_distance_matrices(combined_clips)
            np.testing.assert_allclose(first_distances, combined_distances[:5, :5], atol=1e-12)
            self.assertTrue(np.isnan(combined_distances[0, 5]))

    def test_perceptual_schedule_is_balanced_deterministic_and_nonrepeating(self) -> None:
        positions = np.asarray([0.0, 0.25, 0.55, 0.85, 1.0], dtype=np.float64)
        distances = np.abs(positions[:, None] - positions[None, :])
        first = perceptual_schedule(range(5), distances, count=15, seed=42)
        second = perceptual_schedule(range(5), distances, count=15, seed=42)
        self.assertEqual(first, second)
        self.assertTrue(all(left != right for left, right in zip(first, first[1:])))
        self.assertEqual(set(Counter(first).values()), {3})
        diagnostics = schedule_diagnostics(first, distances)
        self.assertEqual(diagnostics["immediate_repeats"], 0)
        self.assertGreater(diagnostics["mean_adjacent_distance"], 0.0)
        self.assertGreater(diagnostics["transition_entropy"], 2.3)

    def test_content_aware_schedule_does_not_always_start_with_medoid(self) -> None:
        positions = np.asarray([0.0, 0.25, 0.55, 0.85, 1.0], dtype=np.float64)
        distances = np.abs(positions[:, None] - positions[None, :])
        center = medoid_index(range(5), distances)
        starts = {
            perceptual_schedule(range(5), distances, count=15, seed=seed)[0]
            for seed in range(20)
        }
        self.assertGreater(len(starts), 1)
        self.assertTrue(any(value != center for value in starts))

    def test_shuffle_is_balanced_when_event_count_is_multiple_of_pool(self) -> None:
        schedule = shuffle_schedule([10, 11, 12], count=15, seed=17)
        self.assertEqual(Counter(schedule), Counter({10: 5, 11: 5, 12: 5}))
        self.assertTrue(all(left != right for left, right in zip(schedule, schedule[1:])))

    def test_rendering_preserves_stereo_shape_and_peak_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_group(directory, 1, 3)
            clips = analyze_directory(directory, clip_duration_s=1.2)
            distances, _ = build_distance_matrices(clips)
            pool = list(range(3))
            bank = normalize_prepared_bank(clips, pool)
            repeat = assemble_sequence(bank, [0] * 6, self.sample_rate, interval_ms=400.0)
            changed_schedule = perceptual_schedule(pool, distances, count=6, seed=19)
            changed = assemble_sequence(bank, changed_schedule, self.sample_rate, interval_ms=400.0)
            matched = rms_match_sequences({"repeat": repeat, "changed": changed}, anchor="repeat")
            self.assertEqual(matched["repeat"].shape, matched["changed"].shape)
            self.assertEqual(matched["repeat"].shape[1], 2)
            for audio in matched.values():
                passed, failures = technical_audio_gate(audio)
                self.assertTrue(passed, failures)


if __name__ == "__main__":
    unittest.main()
