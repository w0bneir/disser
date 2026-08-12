"""Модульные проверки Direct Latent Guidance без загрузки модели и без GPU."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from stable_audio_guidance import (
    _x0_from_v_prediction,
    active_latent_length,
    envelope_metrics,
    guide_latents,
    latent_rms_envelope,
    resample_target_envelope,
    predict_noise_sequential_cfg,
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
