"""Проверки DSP-части демонстратора без загрузки AudioLDM или GPU."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from audio_io import peak_normalize
from analyzer import extract_rms_envelope
from guided_pipeline import (
    _clip_gradient_norm,
    _decode_waveform,
    envelope_metrics,
    latent_rms_envelope,
    resample_target_envelope,
)


class _WrongShapeDecoder:
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 1, 1, 64, 64)

    def mel_spectrogram_to_waveform(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Вокодер не должен быть вызван при неверной форме VAE-выхода")


class _FakeScheduler:
    def __init__(self) -> None:
        self.alphas_cumprod = torch.tensor([0.9, 0.7], dtype=torch.float32)
        self.timesteps = torch.tensor([1, 0])

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
        self.timesteps = torch.tensor([1, 0], device=device)[:num_inference_steps]

    def scale_model_input(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return latents

    def step(self, noise_prediction: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(prev_sample=latents - 0.01 * noise_prediction)


class _FakeUNet:
    def __call__(self, latents: torch.Tensor, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(sample=torch.zeros_like(latents))


class _FakePipe:
    def __init__(self) -> None:
        self.scheduler = _FakeScheduler()
        self.unet = _FakeUNet()
        self.vocoder = SimpleNamespace(config=SimpleNamespace(sampling_rate=16000))
        self.decode_calls = 0

    def _encode_prompt(self, **kwargs: object) -> torch.Tensor:
        return torch.zeros(2, 512)

    def prepare_extra_step_kwargs(self, generator: torch.Generator, eta: float) -> dict[str, object]:
        return {}

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        return torch.zeros(latents.shape[0], 1, 6, 64)

    def mel_spectrogram_to_waveform(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        return torch.zeros(mel_spectrogram.shape[0], 128)


class DirectLatentGuidanceTests(unittest.TestCase):
    def test_latent_envelope_has_time_axis_and_normalized_range(self) -> None:
        latents = torch.arange(2 * 3 * 7 * 4, dtype=torch.float32).reshape(2, 3, 7, 4)
        envelope = latent_rms_envelope(latents)
        self.assertEqual(tuple(envelope.shape), (2, 7))
        self.assertTrue(torch.all(envelope >= 0))
        self.assertTrue(torch.all(envelope <= 1))
        self.assertTrue(torch.allclose(envelope[:, 0], torch.zeros(2)))
        self.assertTrue(torch.allclose(envelope[:, -1], torch.ones(2)))

    def test_latent_envelope_rejects_wrong_rank(self) -> None:
        with self.assertRaises(ValueError):
            latent_rms_envelope(torch.zeros(1, 4, 8))

    def test_resample_target_envelope(self) -> None:
        target = torch.tensor([0.0, 1.0])
        resampled = resample_target_envelope(target, 5)
        self.assertEqual(tuple(resampled.shape), (5,))
        self.assertAlmostEqual(float(resampled[0]), 0.0, places=6)
        self.assertAlmostEqual(float(resampled[-1]), 1.0, places=6)

    def test_metrics_for_identical_envelopes(self) -> None:
        envelope = torch.tensor([0.0, 0.25, 1.0, 0.5])
        metrics = envelope_metrics(envelope, envelope)
        self.assertAlmostEqual(metrics["mse"], 0.0, places=7)
        self.assertAlmostEqual(metrics["pearson_correlation"], 1.0, places=6)

    def test_gradient_is_clipped_by_l2_norm(self) -> None:
        gradient = torch.full((2, 4, 3, 2), 10.0)
        clipped = _clip_gradient_norm(gradient, max_norm=0.25)
        norms = torch.linalg.vector_norm(clipped.flatten(start_dim=1), dim=1)
        self.assertTrue(torch.all(norms <= 0.250001))

    def test_short_waveform_still_has_one_rms_frame(self) -> None:
        envelope = extract_rms_envelope(torch.ones(64))
        self.assertEqual(envelope.numel(), 1)
        self.assertTrue(torch.allclose(envelope, torch.zeros(1)))

    def test_decoder_output_with_extra_axis_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            _decode_waveform(_WrongShapeDecoder(), torch.zeros(1, 4, 8, 16), 100)

    def test_peak_normalization_prevents_clipping_without_changing_shape(self) -> None:
        normalized = peak_normalize(torch.tensor([-2.0, 0.0, 1.0]).numpy())
        self.assertAlmostEqual(float(abs(normalized).max()), 0.95, places=6)
        self.assertAlmostEqual(float(normalized[2] / normalized[0]), -0.5, places=6)

    def test_guided_denoising_decodes_vae_only_after_the_loop(self) -> None:
        from guided_pipeline import generate_sfx

        pipe = _FakePipe()
        result = generate_sfx(
            pipe,
            prompt="metal impact",
            initial_latents=torch.randn(1, 4, 6, 4),
            original_waveform_length=100,
            target_envelope=torch.tensor([0.0, 1.0, 0.5]),
            seed=17,
            mode="guided",
            gamma=20.0,
            num_inference_steps=2,
        )
        self.assertEqual(pipe.decode_calls, 1)
        self.assertEqual(result.audio.shape, (100,))
        self.assertEqual(len(result.guidance_losses), 2)


if __name__ == "__main__":
    unittest.main()
