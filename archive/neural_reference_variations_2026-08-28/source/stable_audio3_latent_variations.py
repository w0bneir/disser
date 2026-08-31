"""Conservative local perturbations of Stable Audio 3 SAME latents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TangentPerturbationParameters:
    """Parameters for a norm-preserving local latent rotation."""

    angle_degrees: float
    protect_frames: int = 2
    transition_frames: int = 2
    smoothing_frames: int = 3
    covariance_rank: int = 8

    def validate(self, latent_frames: int) -> None:
        if not 0 < self.angle_degrees <= 15:
            raise ValueError("angle_degrees must lie in (0, 15]")
        if not 0 <= self.protect_frames < latent_frames:
            raise ValueError("protect_frames must lie in [0, latent_frames)")
        if self.transition_frames < 0:
            raise ValueError("transition_frames must be non-negative")
        if self.smoothing_frames <= 0 or self.smoothing_frames % 2 == 0:
            raise ValueError("smoothing_frames must be a positive odd integer")
        if self.covariance_rank <= 0:
            raise ValueError("covariance_rank must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def temporal_edit_mask(
    latent_frames: int,
    *,
    protect_frames: int,
    transition_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a [T] mask that protects the event core and fades edits in."""
    if latent_frames <= 0 or not 0 <= protect_frames < latent_frames:
        raise ValueError("Invalid latent/protected frame counts")
    mask = torch.ones(latent_frames, device=device, dtype=dtype)
    mask[:protect_frames] = 0
    fade_length = min(transition_frames, latent_frames - protect_frames)
    if fade_length > 0:
        phase = torch.linspace(
            0,
            pi,
            fade_length + 2,
            device=device,
            dtype=dtype,
        )[1:-1]
        mask[protect_frames : protect_frames + fade_length] = 0.5 - 0.5 * torch.cos(phase)
    return mask


def _smooth_time(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size == 1:
        return values
    padding = kernel_size // 2
    padded = F.pad(values, (padding, padding), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)


def tangent_covariance_rotation(
    latents: torch.Tensor,
    *,
    parameters: TangentPerturbationParameters,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Rotate each latent frame without changing its L2 norm.

    Directions are sampled only from the temporal covariance subspace of the
    reference latent.  This is deliberately more conservative than isotropic
    Gaussian noise: it avoids directions unsupported by the single example.
    """
    if latents.ndim != 3 or latents.shape[0] != 1:
        raise ValueError("Expected latents with shape [1, channels, frames]")
    if not torch.isfinite(latents).all():
        raise ValueError("Latents contain NaN or Inf")
    frames = latents.shape[-1]
    parameters.validate(frames)

    working = latents.detach().float()
    centered = working[0] - working[0].mean(dim=-1, keepdim=True)
    basis, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    usable_rank = min(
        parameters.covariance_rank,
        basis.shape[-1],
        max(1, frames - 1),
    )
    basis = basis[:, :usable_rank]

    generator = torch.Generator(device=working.device)
    generator.manual_seed(int(seed))
    coefficients = torch.randn(
        1,
        usable_rank,
        frames,
        generator=generator,
        device=working.device,
        dtype=working.dtype,
    )
    coefficients = _smooth_time(coefficients, parameters.smoothing_frames)
    direction = torch.einsum("cr,brt->bct", basis, coefficients)

    frame_norm = torch.linalg.vector_norm(working, dim=1, keepdim=True).clamp_min(1e-8)
    unit_latent = working / frame_norm
    direction = direction - (direction * unit_latent).sum(dim=1, keepdim=True) * unit_latent
    direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-8)
    unit_direction = direction / direction_norm

    edit_mask = temporal_edit_mask(
        frames,
        protect_frames=parameters.protect_frames,
        transition_frames=parameters.transition_frames,
        device=working.device,
        dtype=working.dtype,
    )[None, None]
    angle = edit_mask * (parameters.angle_degrees * pi / 180.0)
    rotated = frame_norm * (torch.cos(angle) * unit_latent + torch.sin(angle) * unit_direction)
    rotated = torch.where(edit_mask == 0, working, rotated)
    rotated = rotated.to(dtype=latents.dtype)

    original_norm = torch.linalg.vector_norm(working, dim=1)
    rotated_norm = torch.linalg.vector_norm(rotated.float(), dim=1)
    norm_relative_error = float(
        ((rotated_norm - original_norm).abs() / original_norm.clamp_min(1e-8)).max().cpu()
    )
    delta = rotated.float() - working
    diagnostics: dict[str, float | int] = {
        **parameters.to_dict(),
        "seed": int(seed),
        "latent_channels": int(latents.shape[1]),
        "latent_frames": int(frames),
        "usable_covariance_rank": int(usable_rank),
        "leading_singular_fraction": float(
            singular_values[0].square()
            / singular_values.square().sum().clamp_min(1e-12)
        ),
        "relative_delta_l2": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(working).clamp_min(1e-8)
        ),
        "max_frame_norm_relative_error": norm_relative_error,
    }
    if not torch.isfinite(rotated).all():
        raise FloatingPointError("Perturbed latents contain NaN or Inf")
    return rotated, diagnostics
