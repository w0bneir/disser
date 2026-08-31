"""Task-aligned coarse VampNet LoRA pilot for prompt-free SFX variations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_vampnet_reference_variations import require_safe_gpu
from vampnet_reference_variations import validate_model_assets


TRAINING_PROTOCOL_ID = "sfx_lora_coarse_cb1_3_reconstruction_v1"


def build_task_aligned_mask(
    shape: tuple[int, int, int],
    *,
    attack_tokens: int,
    anchor_periods: np.ndarray,
    anchor_offsets: np.ndarray,
) -> np.ndarray:
    """Mask CB1-3 only, preserving CB0, attack and per-item time anchors."""
    batch, codebooks, steps = shape
    if batch < 1 or codebooks != 4 or steps < 1:
        raise ValueError("Task-aligned coarse mask expects shape (batch, 4, time)")
    if attack_tokens < 0:
        raise ValueError("attack_tokens не может быть отрицательным")
    if anchor_periods.shape != (batch,) or anchor_offsets.shape != (batch,):
        raise ValueError("Нужен один anchor period/offset на batch item")
    if np.any(anchor_periods < 2):
        raise ValueError("Anchor periods должны быть не меньше 2")

    mask = np.zeros(shape, dtype=np.int64)
    mask[:, 1:4, :] = 1
    for index in range(batch):
        period = int(anchor_periods[index])
        offset = int(anchor_offsets[index]) % period
        mask[index, 1:4, offset::period] = 0
    if attack_tokens:
        mask[:, :, : min(attack_tokens, steps)] = 0
    return mask


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_validation_batches(
    indices: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    if indices.size == 0:
        raise ValueError("Validation split пуст")
    rng = np.random.default_rng(seed)
    size = min(count, indices.size)
    return np.sort(rng.choice(indices, size=size, replace=False))


def train_pilot(
    *,
    token_cache: Path,
    model_dir: Path,
    output_dir: Path,
    steps: int,
    batch_size: int,
    learning_rate: float,
    validation_every: int,
    validation_items: int,
    seed: int,
    attack_tokens: int,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Каталог LoRA run не пуст: {output_dir}")
    if not token_cache.is_file():
        raise FileNotFoundError(f"Не найден token cache: {token_cache}")
    if not 1 <= steps <= 500:
        raise ValueError("steps должны быть в диапазоне 1..500")
    if not 1 <= batch_size <= 8:
        raise ValueError("batch_size должен быть в диапазоне 1..8")
    if not 1e-6 <= learning_rate <= 5e-4:
        raise ValueError("learning_rate вне безопасного диапазона")
    if validation_every < 1 or validation_items < 1:
        raise ValueError("Некорректные validation параметры")
    assets = validate_model_assets(model_dir, required=("codec.pth", "coarse.pth"))
    gpu = require_safe_gpu()

    archive = np.load(token_cache)
    required_arrays = {"codes", "train_indices", "val_indices"}
    if not required_arrays.issubset(archive.files):
        raise ValueError("Token cache не содержит обязательные arrays")
    all_codes = archive["codes"]
    train_indices = archive["train_indices"].astype(np.int64, copy=False)
    val_indices = archive["val_indices"].astype(np.int64, copy=False)
    if all_codes.ndim != 3 or all_codes.shape[1] != 14:
        raise ValueError(f"Некорректная форма cached codes: {all_codes.shape}")
    if int(all_codes.min()) < 0 or int(all_codes.max()) >= 1024:
        raise ValueError("Cached token id вне vocab 0..1023")
    coarse_codes = all_codes[:, :4, :].astype(np.int64, copy=False)

    import loralib as lora
    from lac.model.lac import LAC
    from vampnet.modules.transformer import VampNet
    from vampnet.util import codebook_flatten

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    torch.set_float32_matmul_precision("high")
    print("[+] Загрузка codec + coarse VampNet для task-aligned LoRA...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    codec = LAC.load(model_dir / "codec.pth", map_location="cpu")
    codec.eval().requires_grad_(False).to("cuda")
    model = VampNet.load(model_dir / "coarse.pth", map_location="cpu", strict=False)
    lora.mark_only_lora_as_trainable(model)
    model.to("cuda")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    def make_mask(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        periods = rng.integers(5, 10, size=codes.shape[0], dtype=np.int64)
        offsets = np.asarray(
            [rng.integers(0, int(period)) for period in periods],
            dtype=np.int64,
        )
        return build_task_aligned_mask(
            tuple(codes.shape),
            attack_tokens=attack_tokens,
            anchor_periods=periods,
            anchor_offsets=offsets,
        )

    def loss_for_batch(codes_np: np.ndarray, mask_np: np.ndarray) -> torch.Tensor:
        codes = torch.from_numpy(codes_np).to(device="cuda", dtype=torch.long)
        mask = torch.from_numpy(mask_np).to(device="cuda", dtype=torch.long)
        masked_codes = codes.masked_fill(mask.bool(), model.mask_token)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = model.embedding.from_codes(masked_codes, codec)
            logits = model(latents)
            target = codebook_flatten(codes)
            flat_mask = codebook_flatten(mask)
            target = target.masked_fill(~flat_mask.bool(), -100)
            loss = F.cross_entropy(logits, target, label_smoothing=0.05)
        return loss

    fixed_val_indices = _fixed_validation_batches(
        val_indices,
        count=validation_items,
        seed=seed + 10_000,
    )
    fixed_val_codes = coarse_codes[fixed_val_indices]
    fixed_val_mask = make_mask(
        fixed_val_codes,
        np.random.default_rng(seed + 20_000),
    )

    @torch.no_grad()
    def validation_loss() -> float:
        model.eval()
        losses = []
        for start in range(0, fixed_val_codes.shape[0], batch_size):
            batch_codes = fixed_val_codes[start : start + batch_size]
            batch_mask = fixed_val_mask[start : start + batch_size]
            losses.append(float(loss_for_batch(batch_codes, batch_mask).float().cpu()))
        return float(np.mean(losses))

    started = time.perf_counter()
    base_validation_loss = validation_loss()
    print(f"[+] Base validation loss: {base_validation_loss:.4f}", flush=True)
    train_rows: list[dict[str, float | int]] = []
    validation_rows: list[dict[str, float | int]] = [
        {"step": 0, "loss": base_validation_loss}
    ]
    order = np_rng.permutation(train_indices)
    cursor = 0
    for step in range(1, steps + 1):
        if cursor + batch_size > order.size:
            order = np_rng.permutation(train_indices)
            cursor = 0
        batch_indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        batch_codes = coarse_codes[batch_indices]
        batch_mask = make_mask(batch_codes, np_rng)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_batch(batch_codes, batch_mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Нефинитный training loss на шаге {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Нефинитный gradient norm на шаге {step}")
        optimizer.step()
        train_rows.append(
            {
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "gradient_norm": float(grad_norm.detach().float().cpu()),
                "masked_fraction": float(batch_mask.mean()),
            }
        )
        if step == 1 or step % validation_every == 0 or step == steps:
            current_validation = validation_loss()
            validation_rows.append({"step": step, "loss": current_validation})
            print(
                f"    step={step}/{steps}; train={train_rows[-1]['loss']:.4f}; "
                f"val={current_validation:.4f}; grad={train_rows[-1]['gradient_norm']:.3f}",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    lora_path = output_dir / "coarse_lora.pth"
    torch.save(lora.lora_state_dict(model), lora_path)
    final_validation_loss = float(validation_rows[-1]["loss"])
    report = {
        "stage": "vampnet_task_aligned_sfx_lora_pilot",
        "protocol_id": TRAINING_PROTOCOL_ID,
        "token_cache": str(token_cache.resolve()),
        "model_assets": assets,
        "gpu": gpu,
        "configuration": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_every": validation_every,
            "validation_items": int(fixed_val_indices.size),
            "seed": seed,
            "attack_tokens": attack_tokens,
            "masked_codebooks": [1, 2, 3],
            "preserved_codebooks": [0],
            "anchor_period_range_inclusive": [5, 9],
            "optimizer": "AdamW beta=(0.9,0.95), weight_decay=0.01",
            "precision": "bfloat16 autocast",
        },
        "parameters": {
            "total": total_count,
            "trainable_lora": trainable_count,
            "trainable_fraction": trainable_count / total_count,
        },
        "base_validation_loss": base_validation_loss,
        "final_validation_loss": final_validation_loss,
        "validation_improvement_fraction": (
            (base_validation_loss - final_validation_loss) / max(base_validation_loss, 1e-12)
        ),
        "training": train_rows,
        "validation": validation_rows,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "lora_checkpoint": {
            "path": str(lora_path.resolve()),
            "bytes": lora_path.stat().st_size,
            "sha256": _sha256(lora_path),
        },
        "requires_generation_and_human_listening_gate": True,
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[+] LoRA pilot: val {base_validation_loss:.4f} -> {final_validation_loss:.4f}; "
        f"peak={report['peak_vram_mib']:.0f} MiB",
        flush=True,
    )
    print(f"[+] Checkpoint: {lora_path.resolve()}", flush=True)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task-aligned coarse VampNet SFX LoRA")
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/vampnet_models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--validation-items", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--attack-tokens", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        train_pilot(
            token_cache=args.token_cache,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_every=args.validation_every,
            validation_items=args.validation_items,
            seed=args.seed,
            attack_tokens=args.attack_tokens,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[!] LoRA training blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
