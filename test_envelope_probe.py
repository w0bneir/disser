"""CPU-проверки waveform-aware envelope probe."""

from __future__ import annotations

import unittest

import torch

from envelope_probe import WaveformEnvelopeProbe, envelope_training_loss
from train_envelope_probe import ProbeSample, split_probe_samples


class EnvelopeProbeTests(unittest.TestCase):
    def test_probe_output_shape_range_and_gradient(self) -> None:
        torch.manual_seed(1)
        latents = torch.randn(2, 4, 11, requires_grad=True)
        probe = WaveformEnvelopeProbe(latent_channels=4, temporal_kernel_size=5)
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


if __name__ == "__main__":
    unittest.main()
