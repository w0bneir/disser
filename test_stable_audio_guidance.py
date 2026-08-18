"""Модульные проверки Direct Latent Guidance без загрузки модели и без GPU."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from analyzer import load_audio
from envelope_probe import WaveformEnvelopeProbe
from run_stable_audio_experiments import (
    PAIR_OUTPUT_FILES,
    LATENT_DIAGNOSTICS_FILE,
    load_reference_for_analysis,
    pair_is_complete,
    resolve_experiment_selection,
    resolve_guidance_gamma,
    resolve_inference_steps,
    save_latent_diagnostics,
    validate_envelope_probe_request,
    validate_latent_diagnostics_request,
    validate_probe_guidance_mode,
)
from stable_audio_guidance import (
    _x0_from_v_prediction,
    active_latent_length,
    envelope_metrics,
    guide_final_latents,
    guide_final_latents_with_decoder,
    guide_latents,
    latent_rms_envelope,
    predict_guidance_envelope,
    resample_target_envelope,
    predict_noise_sequential_cfg,
    waveform_rms_envelope,
)


class _PipeShape:
    vae = SimpleNamespace(config=SimpleNamespace(sampling_rate=44_100), hop_length=2048)
    transformer = SimpleNamespace(config=SimpleNamespace(sample_size=1024))


class _Scheduler:
    def scale_model_input(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return latents + 1


class _Transformer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, latents: torch.Tensor, timestep: torch.Tensor, *, encoder_hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
        self.batch_sizes.append(latents.shape[0])
        return (latents + encoder_hidden_states[:, :1, :1].reshape(-1, 1, 1),)


class _CfgPipe:
    def __init__(self) -> None:
        self.scheduler = _Scheduler()
        self.transformer = _Transformer()


class StableAudioGuidanceTests(unittest.TestCase):
    def test_final_probe_mode_requires_probe_and_valid_steps(self) -> None:
        validate_probe_guidance_mode(
            probe_guidance_mode="denoising",
            final_guidance_steps=10,
            envelope_probe_path=None,
        )
        with self.assertRaisesRegex(ValueError, "требует --envelope-probe"):
            validate_probe_guidance_mode(
                probe_guidance_mode="final",
                final_guidance_steps=10,
                envelope_probe_path=None,
            )
        with self.assertRaisesRegex(ValueError, "диапазоне"):
            validate_probe_guidance_mode(
                probe_guidance_mode="denoising",
                final_guidance_steps=0,
                envelope_probe_path=None,
            )
        validate_probe_guidance_mode(
            probe_guidance_mode="decoder",
            final_guidance_steps=3,
            envelope_probe_path=Path("probe.safetensors"),
        )
        with self.assertRaisesRegex(ValueError, "диапазоне"):
            validate_probe_guidance_mode(
                probe_guidance_mode="decoder",
                final_guidance_steps=4,
                envelope_probe_path=Path("probe.safetensors"),
            )

    def test_envelope_probe_request_is_guarded(self) -> None:
        validate_envelope_probe_request(envelope_probe_path=None, max_new_pairs=None)
        with TemporaryDirectory() as directory:
            weights = Path(directory) / "probe.safetensors"
            metadata = weights.with_suffix(".json")
            weights.touch()
            metadata.touch()
            validate_envelope_probe_request(
                envelope_probe_path=weights,
                max_new_pairs=1,
            )
            with self.assertRaisesRegex(ValueError, "--max-new-pairs 1"):
                validate_envelope_probe_request(
                    envelope_probe_path=weights,
                    max_new_pairs=None,
                )

    def test_resume_requires_complete_pair_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "baseline.wav").touch()
            (run_dir / "guided.wav").touch()
            self.assertFalse(pair_is_complete(run_dir))
            for name in PAIR_OUTPUT_FILES:
                (run_dir / name).touch()
            self.assertTrue(pair_is_complete(run_dir))
            self.assertFalse(pair_is_complete(run_dir, require_latent_diagnostics=True))
            (run_dir / LATENT_DIAGNOSTICS_FILE).touch()
            self.assertTrue(pair_is_complete(run_dir, require_latent_diagnostics=True))

    def test_latent_diagnostics_request_requires_one_pair(self) -> None:
        validate_latent_diagnostics_request(
            export_latent_diagnostics=False,
            max_new_pairs=None,
        )
        validate_latent_diagnostics_request(
            export_latent_diagnostics=True,
            max_new_pairs=1,
        )
        with self.assertRaisesRegex(ValueError, "--max-new-pairs 1"):
            validate_latent_diagnostics_request(
                export_latent_diagnostics=True,
                max_new_pairs=None,
            )

    def test_latent_diagnostics_npz_is_numeric_and_aligned(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / LATENT_DIAGNOSTICS_FILE
            target = torch.tensor([0.0, 0.5, 1.0, 0.0])
            baseline_latent = torch.tensor([0.0, 0.3, 1.0])
            guided_latent = torch.tensor([0.0, 0.6, 1.0])
            baseline_waveform = torch.tensor([0.0, 0.2, 0.7, 1.0, 0.2])
            guided_waveform = torch.tensor([0.0, 0.4, 0.9, 1.0, 0.1])
            baseline_active = torch.arange(12, dtype=torch.float16).reshape(4, 3)
            guided_active = baseline_active + 0.25

            metadata = save_latent_diagnostics(
                output,
                target_envelope=target,
                baseline_latent_envelope=baseline_latent,
                guided_latent_envelope=guided_latent,
                baseline_waveform_envelope=baseline_waveform,
                guided_waveform_envelope=guided_waveform,
                baseline_active_latents=baseline_active,
                guided_active_latents=guided_active,
                sample_rate=44_100,
                duration_seconds=0.5,
                latent_hop_length=2_048,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(metadata["active_latent_shape"], [4, 3])
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive["baseline_active_latents"].shape, (4, 3))
                self.assertEqual(archive["target_envelope_latent"].shape, (3,))
                self.assertEqual(archive["target_envelope_waveform"].shape, (5,))
                self.assertEqual(archive["format_version"].item(), 1)
                self.assertEqual(archive["latent_hop_length"].item(), 2_048)

    def test_reference_wav_loads_as_16khz_mono_without_librosa(self) -> None:
        reference = Path(__file__).parent / "references" / "metal_impact.wav"
        waveform, sample_rate = load_audio(reference)
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(waveform.ndim, 1)
        self.assertGreater(waveform.numel(), 0)
        self.assertTrue(torch.isfinite(waveform).all())

    def test_reference_analysis_uses_model_sample_rate(self) -> None:
        reference = Path(__file__).parent / "references" / "metal_impact.wav"
        waveform, sample_rate, envelope = load_reference_for_analysis(
            reference,
            analysis_sample_rate=44_100,
        )
        self.assertEqual(sample_rate, 44_100)
        expected_points = 1 + (waveform.numel() - 2048) // 512
        self.assertEqual(envelope.numel(), expected_points)
        self.assertEqual(envelope.numel(), 37)
        self.assertTrue(torch.isfinite(envelope).all())

    def test_smoke_test_always_uses_four_steps(self) -> None:
        self.assertEqual(
            resolve_inference_steps(
                configured_steps=50,
                smoke_test=True,
                requested_steps=None,
                max_new_pairs=1,
            ),
            4,
        )

    def test_intermediate_steps_require_one_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "--max-new-pairs 1"):
            resolve_inference_steps(
                configured_steps=50,
                smoke_test=False,
                requested_steps=10,
                max_new_pairs=None,
            )

    def test_intermediate_steps_validate_range_and_smoke_conflict(self) -> None:
        with self.assertRaisesRegex(ValueError, "диапазоне"):
            resolve_inference_steps(
                configured_steps=50,
                smoke_test=False,
                requested_steps=51,
                max_new_pairs=1,
            )
        with self.assertRaisesRegex(ValueError, "нельзя совмещать"):
            resolve_inference_steps(
                configured_steps=50,
                smoke_test=True,
                requested_steps=10,
                max_new_pairs=1,
            )

    def test_gamma_override_requires_one_pair_and_valid_range(self) -> None:
        self.assertEqual(
            resolve_guidance_gamma(
                configured_gamma=20.0,
                requested_gamma=None,
                max_new_pairs=None,
            ),
            20.0,
        )
        self.assertEqual(
            resolve_guidance_gamma(
                configured_gamma=20.0,
                requested_gamma=50.0,
                max_new_pairs=1,
            ),
            50.0,
        )
        with self.assertRaisesRegex(ValueError, "--max-new-pairs 1"):
            resolve_guidance_gamma(
                configured_gamma=20.0,
                requested_gamma=50.0,
                max_new_pairs=None,
            )
        for invalid_gamma in (0.0, 50.1, float("inf"), float("nan")):
            with self.subTest(gamma=invalid_gamma), self.assertRaisesRegex(ValueError, "диапазоне"):
                resolve_guidance_gamma(
                    configured_gamma=20.0,
                    requested_gamma=invalid_gamma,
                    max_new_pairs=1,
                )

    def test_case_and_seed_selection_is_guarded(self) -> None:
        cases = [{"id": "metal"}, {"id": "wood"}]
        selected_cases, selected_seeds = resolve_experiment_selection(
            configured_cases=cases,
            configured_seeds=[17, 42],
            requested_case_id="wood",
            requested_seed=2026,
            smoke_test=False,
            max_new_pairs=1,
        )
        self.assertEqual(selected_cases, [{"id": "wood"}])
        self.assertEqual(selected_seeds, [2026])

        with self.assertRaisesRegex(ValueError, "--max-new-pairs 1"):
            resolve_experiment_selection(
                configured_cases=cases,
                configured_seeds=[17, 42],
                requested_case_id="wood",
                requested_seed=None,
                smoke_test=False,
                max_new_pairs=None,
            )
        with self.assertRaisesRegex(ValueError, "Неизвестный --case-id"):
            resolve_experiment_selection(
                configured_cases=cases,
                configured_seeds=[17, 42],
                requested_case_id="glass",
                requested_seed=None,
                smoke_test=False,
                max_new_pairs=1,
            )

    def test_latent_envelope_shape_and_normalization(self) -> None:
        latents = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
        envelope = latent_rms_envelope(latents, active_length=5)
        self.assertEqual(tuple(envelope.shape), (2, 5))
        self.assertTrue(torch.all(envelope >= 0))
        self.assertTrue(torch.all(envelope <= 1))

    def test_latent_envelope_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            latent_rms_envelope(torch.zeros(1, 4, 8, 2))

    def test_target_resampling(self) -> None:
        result = resample_target_envelope(torch.tensor([0.0, 1.0]), 5)
        self.assertTrue(torch.allclose(result, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])))

    def test_x0_formula_for_v_prediction(self) -> None:
        latents = torch.full((1, 2, 3), 2.0)
        prediction = torch.ones_like(latents)
        result = _x0_from_v_prediction(latents, prediction, 1.0, sigma_data=1.0, prediction_type="v_prediction")
        expected = 0.5 * latents - (1.0 / 2**0.5) * prediction
        self.assertTrue(torch.allclose(result, expected))

    def test_guidance_moves_loss_down_for_linear_case(self) -> None:
        latents = torch.tensor([[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]])
        prediction = torch.zeros_like(latents)
        target = torch.tensor([1.0, 0.5, 0.0])
        before = latent_rms_envelope(latents)[0]
        before_mse = envelope_metrics(target, before)["mse"]
        corrected, _, _, diagnostics = guide_latents(
            latents,
            prediction,
            sigma=0.0,
            target_envelope=target,
            active_length=3,
            gamma=0.1,
            gradient_clip_norm=10.0,
            max_relative_step=1.0,
        )
        after_mse = envelope_metrics(target, latent_rms_envelope(corrected)[0])["mse"]
        self.assertLess(after_mse, before_mse)
        self.assertLess(diagnostics["loss_after"], before_mse)

    def test_probe_guidance_moves_probe_loss_down(self) -> None:
        probe = WaveformEnvelopeProbe(2, ridge_alpha=1.0)
        probe.set_ridge_state(
            feature_mean=torch.zeros(2),
            feature_scale=torch.ones(2),
            channel_weights=torch.tensor([1.0, -0.25]),
            bias=0.0,
        )
        latents = torch.tensor([[[0.1, 0.5, 1.0], [0.3, 0.2, 0.1]]])
        prediction = torch.zeros_like(latents)
        target = torch.tensor([1.0, 0.5, 0.0])
        before = predict_guidance_envelope(
            latents,
            active_length=3,
            envelope_probe=probe,
        )[0]
        before_mse = envelope_metrics(target, before)["mse"]
        corrected, _, _, diagnostics = guide_latents(
            latents,
            prediction,
            sigma=0.0,
            target_envelope=target,
            active_length=3,
            gamma=0.1,
            gradient_clip_norm=10.0,
            max_relative_step=1.0,
            envelope_probe=probe,
        )
        after = predict_guidance_envelope(
            corrected,
            active_length=3,
            envelope_probe=probe,
        )[0]
        after_mse = envelope_metrics(target, after)["mse"]
        self.assertLess(after_mse, before_mse)
        self.assertLess(diagnostics["loss_after"], before_mse)

    def test_final_probe_guidance_enforces_total_per_frame_trust_region(self) -> None:
        probe = WaveformEnvelopeProbe(2, ridge_alpha=1.0)
        probe.set_ridge_state(
            feature_mean=torch.zeros(2),
            feature_scale=torch.ones(2),
            channel_weights=torch.tensor([1.0, -0.25]),
            bias=0.0,
        )
        latents = torch.tensor(
            [[[0.1, 0.5, 1.0, 0.7, 0.2], [0.3, 0.2, 0.1, 0.4, 0.6]]]
        )
        target = torch.tensor([1.0, 0.5, 0.0])
        before = predict_guidance_envelope(
            latents,
            active_length=3,
            envelope_probe=probe,
        )[0]
        corrected, final_loss, trace = guide_final_latents(
            latents,
            target_envelope=target,
            active_length=3,
            envelope_probe=probe,
            gamma=1000.0,
            gradient_clip_norm=1000.0,
            max_relative_step=0.03,
            steps=10,
            reference_active_length=3,
        )
        after = predict_guidance_envelope(
            corrected,
            active_length=3,
            envelope_probe=probe,
        )[0]
        self.assertLess(envelope_metrics(target, after)["mse"], envelope_metrics(target, before)["mse"])
        self.assertAlmostEqual(final_loss, trace[-1]["loss_after"])
        delta = corrected[:, :, :3] - latents[:, :, :3]
        frame_relative = torch.linalg.vector_norm(delta, dim=1) / torch.linalg.vector_norm(
            latents[:, :, :3], dim=1
        )
        self.assertLessEqual(float(frame_relative.max()), 0.030001)
        self.assertTrue(torch.equal(corrected[:, :, 3:], latents[:, :, 3:]))
        self.assertTrue(
            all(float(row["max_frame_relative_correction"]) <= 0.030001 for row in trace)
        )

    def test_decoder_guidance_uses_exact_waveform_loss_inside_trust_region(self) -> None:
        latents = torch.tensor(
            [[[0.1, 0.5, 1.0, 0.7, 0.2], [0.3, 0.2, 0.1, 0.4, 0.6]]]
        )
        target = torch.tensor([1.0, 0.5, 0.0])

        def decode(values: torch.Tensor) -> torch.Tensor:
            return values[:, :1].repeat_interleave(1024, dim=-1)

        before = waveform_rms_envelope(decode(latents[:, :, :3]))[0]
        target_resampled = resample_target_envelope(target, before.numel())
        corrected, final_loss, trace = guide_final_latents_with_decoder(
            latents,
            decode_fn=decode,
            target_envelope=target,
            active_length=3,
            waveform_length=3072,
            gamma=50.0,
            gradient_clip_norm=1000.0,
            max_relative_step=0.03,
            steps=3,
            decoder_context_frames=0,
            reference_active_length=3,
        )
        after = waveform_rms_envelope(decode(corrected[:, :, :3]))[0]
        self.assertLess(
            envelope_metrics(target_resampled, after)["mse"],
            envelope_metrics(target_resampled, before)["mse"],
        )
        self.assertAlmostEqual(final_loss, trace[-1]["loss_after"])
        self.assertEqual(len(trace), 3)
        self.assertTrue(
            all(
                float(current["loss_after"]) <= float(current["loss_before"])
                for current in trace
            )
        )
        delta = corrected[:, :, :3] - latents[:, :, :3]
        frame_relative = torch.linalg.vector_norm(delta, dim=1) / torch.linalg.vector_norm(
            latents[:, :, :3], dim=1
        )
        self.assertLessEqual(float(frame_relative.max()), 0.030001)
        self.assertTrue(
            all(float(row["max_frame_relative_correction"]) <= 0.030001 for row in trace)
        )
        self.assertTrue(torch.equal(corrected[:, :, 3:], latents[:, :, 3:]))

    def test_fp16_early_sigma_stays_finite(self) -> None:
        latents = torch.zeros((1, 64, 11), dtype=torch.float16)
        prediction = torch.zeros_like(latents)
        corrected, envelope, _, diagnostics = guide_latents(
            latents,
            prediction,
            sigma=500.0,
            target_envelope=torch.linspace(0, 1, 11),
            active_length=11,
            gamma=0.5,
            gradient_clip_norm=0.05,
            max_relative_step=0.03,
        )
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertTrue(torch.isfinite(envelope).all())
        self.assertLessEqual(diagnostics["relative_correction"], 0.030001)

    def test_probe_fp16_early_sigma_stays_finite(self) -> None:
        generator = torch.Generator().manual_seed(17)
        latents = torch.randn((1, 64, 11), generator=generator, dtype=torch.float16)
        prediction = torch.randn((1, 64, 11), generator=generator, dtype=torch.float16)
        probe = WaveformEnvelopeProbe(64, ridge_alpha=1.0)
        corrected, envelope, _, diagnostics = guide_latents(
            latents,
            prediction,
            sigma=500.0,
            target_envelope=torch.linspace(0, 1, 11),
            active_length=11,
            gamma=50.0,
            gradient_clip_norm=0.1,
            max_relative_step=0.03,
            envelope_probe=probe,
        )
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertTrue(torch.isfinite(envelope).all())
        self.assertLessEqual(diagnostics["relative_correction"], 0.030001)

    def test_relative_step_cap_is_enforced(self) -> None:
        latents = torch.tensor([[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]])
        corrected, _, _, diagnostics = guide_latents(
            latents,
            torch.zeros_like(latents),
            sigma=0.0,
            target_envelope=torch.tensor([1.0, 0.5, 0.0]),
            active_length=3,
            gamma=1000.0,
            gradient_clip_norm=1000.0,
            max_relative_step=0.03,
        )
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertLessEqual(diagnostics["relative_correction"], 0.030001)

    def test_duration_scaling_keeps_relative_correction_comparable(self) -> None:
        short_latents = torch.tensor([[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]])
        short_target = torch.tensor([1.0, 0.5, 0.0])
        long_latents = short_latents.repeat(1, 1, 2)
        long_target = short_target.repeat(2)

        _, _, _, short_diagnostics = guide_latents(
            short_latents,
            torch.zeros_like(short_latents),
            sigma=0.0,
            target_envelope=short_target,
            active_length=3,
            gamma=0.01,
            gradient_clip_norm=100.0,
            max_relative_step=1.0,
            reference_active_length=3,
        )
        _, _, _, long_diagnostics = guide_latents(
            long_latents,
            torch.zeros_like(long_latents),
            sigma=0.0,
            target_envelope=long_target,
            active_length=6,
            gamma=0.01,
            gradient_clip_norm=100.0,
            max_relative_step=1.0,
            reference_active_length=3,
        )

        self.assertEqual(short_diagnostics["duration_scale"], 1.0)
        self.assertEqual(long_diagnostics["duration_scale"], 2.0)
        self.assertAlmostEqual(
            short_diagnostics["relative_correction"],
            long_diagnostics["relative_correction"],
            places=6,
        )

    def test_active_latent_length_uses_decoder_hop(self) -> None:
        self.assertEqual(active_latent_length(_PipeShape(), 0.47), 11)
        self.assertEqual(active_latent_length(_PipeShape(), 1000), 1024)

    def test_sequential_cfg_never_uses_batch_two(self) -> None:
        pipe = _CfgPipe()
        output = predict_noise_sequential_cfg(
            pipe,
            latents=torch.zeros(1, 2, 3),
            timestep=torch.tensor(1),
            text_duration=torch.tensor([[[2.0]], [[5.0]]]),
            global_duration=torch.zeros(2, 1, 1),
            rotary_embedding=torch.zeros(1),
            do_cfg=True,
            guidance_scale=3.0,
        )
        self.assertEqual(pipe.transformer.batch_sizes, [1, 1])
        self.assertTrue(torch.allclose(output, torch.full((1, 2, 3), 12.0)))


if __name__ == "__main__":
    unittest.main()
