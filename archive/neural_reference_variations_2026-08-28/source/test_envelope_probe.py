"""CPU-проверки waveform-aware envelope probe."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from envelope_probe import (
    WaveformEnvelopeProbe,
    envelope_training_loss,
    load_waveform_envelope_probe,
    normalize_envelope,
)
from train_envelope_probe import ProbeSample, split_probe_samples, train_probe


class EnvelopeProbeTests(unittest.TestCase):
    def test_probe_output_shape_range_and_gradient(self) -> None:
        torch.manual_seed(1)
        latents = torch.randn(2, 4, 11, requires_grad=True)
        probe = WaveformEnvelopeProbe(latent_channels=4)
        predicted = probe(latents, active_length=9)
        self.assertEqual(predicted.shape, (2, 9))
        self.assertTrue(torch.all(predicted >= 0))
        self.assertTrue(torch.all(predicted <= 1))
        predicted.square().mean().backward()
        self.assertIsNotNone(latents.grad)
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(latents.grad)), 0)

    def test_training_loss_rewards_matching_shape(self) -> None:
        target = torch.tensor([[0.0, 0.2, 1.0, 0.3]])
        good_loss, good_metrics = envelope_training_loss(target.clone(), target)
        bad = torch.flip(target, dims=(-1,))
        bad_loss, _ = envelope_training_loss(bad, target)
        self.assertAlmostEqual(float(good_loss), 0.0, places=6)
        self.assertAlmostEqual(float(good_metrics["pearson_correlation"]), 1.0, places=6)
        self.assertGreater(float(bad_loss), float(good_loss))

    def test_group_split_keeps_pair_together(self) -> None:
        samples = []
        for group in ("pair_a", "pair_b", "pair_c", "pair_d"):
            for mode in ("baseline", "guided"):
                samples.append(
                    ProbeSample(
                        name=f"{group}/{mode}",
                        group=group,
                        latents=torch.zeros(4, 5),
                        waveform_envelope=torch.zeros(5),
                    )
                )
        training, validation = split_probe_samples(
            samples,
            validation_fraction=0.25,
            seed=17,
        )
        training_groups = {sample.group for sample in training}
        validation_groups = {sample.group for sample in validation}
        self.assertFalse(training_groups & validation_groups)
        self.assertEqual(len(training) + len(validation), len(samples))

    def test_signed_ridge_recovers_waveform_projection(self) -> None:
        generator = torch.Generator().manual_seed(7)
        samples = []
        for index in range(5):
            latents = torch.randn(3, 20, generator=generator)
            target = normalize_envelope(
                (0.8 * latents[0] - 0.4 * latents[1] + 0.2 * latents[2]).unsqueeze(0)
            )[0]
            samples.append(
                ProbeSample(
                    name=f"pair_{index}/baseline",
                    group=f"pair_{index}",
                    latents=latents,
                    waveform_envelope=target,
                )
            )
        probe, report = train_probe(samples[:4], samples[4:], ridge_alphas=[0.01])
        validation = report["validation"]
        self.assertGreater(float(validation["probe_pearson"]), 0.99)
        self.assertLess(float(validation["probe_mse"]), 1e-3)
        self.assertEqual(probe.config()["architecture"], "signed_latent_ridge_v2")

    def test_safetensors_json_round_trip(self) -> None:
        from safetensors.torch import save_file

        probe = WaveformEnvelopeProbe(3, ridge_alpha=1.0)
        probe.set_ridge_state(
            feature_mean=torch.tensor([0.1, 0.2, 0.3]),
            feature_scale=torch.tensor([1.0, 2.0, 3.0]),
            channel_weights=torch.tensor([0.5, -0.25, 0.75]),
            bias=0.1,
        )
        latents = torch.randn(1, 3, 7)
        with TemporaryDirectory() as directory:
            weights_path = Path(directory) / "probe.safetensors"
            save_file(dict(probe.state_dict()), str(weights_path))
            weights_path.with_suffix(".json").write_text(
                json.dumps({"format_version": 2, "probe": probe.config()}),
                encoding="utf-8",
            )
            loaded = load_waveform_envelope_probe(weights_path)
            torch.testing.assert_close(loaded(latents), probe(latents))


if __name__ == "__main__":
    unittest.main()
