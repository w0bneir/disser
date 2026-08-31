import unittest

import torch

from stable_audio3_latent_variations import (
    TangentPerturbationParameters,
    tangent_covariance_rotation,
    temporal_edit_mask,
)


class StableAudio3LatentVariationTests(unittest.TestCase):
    def test_mask_protects_prefix_and_fades_in(self) -> None:
        mask = temporal_edit_mask(
            8,
            protect_frames=2,
            transition_frames=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(mask[:2], torch.zeros(2)))
        self.assertGreater(float(mask[2]), 0)
        self.assertLess(float(mask[2]), float(mask[3]))
        self.assertTrue(torch.equal(mask[4:], torch.ones(4)))

    def test_rotation_is_deterministic_and_preserves_frame_norms(self) -> None:
        generator = torch.Generator().manual_seed(11)
        latents = torch.randn(1, 16, 12, generator=generator)
        parameters = TangentPerturbationParameters(
            angle_degrees=3,
            protect_frames=2,
            transition_frames=2,
            smoothing_frames=3,
            covariance_rank=5,
        )
        first, first_info = tangent_covariance_rotation(
            latents, parameters=parameters, seed=17
        )
        second, _ = tangent_covariance_rotation(latents, parameters=parameters, seed=17)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(
            torch.linalg.vector_norm(first.float(), dim=1),
            torch.linalg.vector_norm(latents.float(), dim=1),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertTrue(torch.equal(first[:, :, :2], latents[:, :, :2]))
        self.assertGreater(first_info["relative_delta_l2"], 0)
        self.assertLess(first_info["max_frame_norm_relative_error"], 1e-5)

    def test_different_seeds_change_edited_frames(self) -> None:
        latents = torch.randn(1, 12, 10, generator=torch.Generator().manual_seed(3))
        parameters = TangentPerturbationParameters(
            angle_degrees=2,
            protect_frames=1,
            transition_frames=0,
            smoothing_frames=1,
            covariance_rank=4,
        )
        first, _ = tangent_covariance_rotation(latents, parameters=parameters, seed=17)
        second, _ = tangent_covariance_rotation(latents, parameters=parameters, seed=42)
        self.assertFalse(torch.equal(first[:, :, 1:], second[:, :, 1:]))


if __name__ == "__main__":
    unittest.main()
