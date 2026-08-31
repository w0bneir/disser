"""Один forward/backward без optimizer step для оценки SFX-LoRA на RTX 5070."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_vampnet_reference_variations import require_safe_gpu
from vampnet_reference_variations import validate_model_assets


def probe_model(
    *,
    checkpoint: Path,
    codes_np: np.ndarray,
    codec: torch.nn.Module,
    seed: int,
) -> dict[str, float | int | str]:
    import loralib as lora
    from vampnet.modules.transformer import VampNet
    from vampnet.util import codebook_flatten

    started = time.perf_counter()
    model = VampNet.load(checkpoint, map_location="cpu", strict=False)
    lora.mark_only_lora_as_trainable(model)
    model.train().to("cuda")
    codes = torch.from_numpy(codes_np).to(device="cuda", dtype=torch.long)
    codes = codes[:, : model.n_codebooks, :]

    generator = torch.Generator(device="cuda").manual_seed(seed)
    mask = (torch.rand(codes.shape, generator=generator, device="cuda") < 0.5).long()
    mask[:, : model.n_conditioning_codebooks, :] = 0
    masked_codes = codes.masked_fill(mask.bool(), model.mask_token)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latents = model.embedding.from_codes(masked_codes, codec)
        logits = model(latents)
        target = codebook_flatten(codes[:, model.n_conditioning_codebooks :, :])
        flat_mask = codebook_flatten(mask[:, model.n_conditioning_codebooks :, :])
        target = target.masked_fill(~flat_mask.bool(), -100)
        loss = F.cross_entropy(logits, target)
    loss.backward()
    torch.cuda.synchronize()
    grad_square_sum = torch.zeros((), device="cuda")
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_square_sum += parameter.grad.float().square().sum()
    result: dict[str, float | int | str] = {
        "checkpoint": checkpoint.name,
        "n_codebooks": int(model.n_codebooks),
        "conditioning_codebooks": int(model.n_conditioning_codebooks),
        "sequence_tokens": int(codes.shape[-1]),
        "total_parameters": int(total),
        "trainable_lora_parameters": int(trainable),
        "trainable_fraction": float(trainable / total),
        "loss": float(loss.detach().float().cpu()),
        "gradient_norm": float(grad_square_sum.sqrt().detach().cpu()),
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    del loss, logits, latents, target, flat_mask, masked_codes, mask, codes, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_preflight(
    *,
    token_diagnostics: Path,
    model_dir: Path,
    results_dir: Path,
    seed: int,
) -> Path:
    if results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError(f"Каталог результата не пуст: {results_dir}")
    assets = validate_model_assets(
        model_dir,
        required=("codec.pth", "coarse.pth", "c2f.pth"),
    )
    gpu = require_safe_gpu()
    archive = np.load(token_diagnostics)
    if "reference_codes" not in archive:
        raise ValueError("В token diagnostics нет reference_codes")
    codes_np = archive["reference_codes"]

    from lac.model.lac import LAC

    codec = LAC.load(model_dir / "codec.pth", map_location="cpu")
    codec.eval().requires_grad_(False).to("cuda")
    rows = []
    for filename in ("coarse.pth", "c2f.pth"):
        print(f"[+] LoRA dry backward: {filename}...", flush=True)
        row = probe_model(
            checkpoint=model_dir / filename,
            codes_np=codes_np,
            codec=codec,
            seed=seed,
        )
        rows.append(row)
        print(
            f"    loss={row['loss']:.4f}; grad={row['gradient_norm']:.3f}; "
            f"peak={row['peak_vram_mib']:.0f} MiB",
            flush=True,
        )

    report = {
        "stage": "vampnet_lora_forward_backward_preflight",
        "optimizer_step_performed": False,
        "weights_modified_on_disk": False,
        "token_diagnostics": str(token_diagnostics.resolve()),
        "model_assets": assets,
        "gpu": gpu,
        "models": rows,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "lora_preflight.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] LoRA preflight: {path.resolve()}", flush=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VampNet LoRA feasibility preflight")
    parser.add_argument("--token-diagnostics", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/vampnet_models"))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_preflight(
            token_diagnostics=args.token_diagnostics,
            model_dir=args.model_dir,
            results_dir=args.results_dir,
            seed=args.seed,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[!] LoRA preflight blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
