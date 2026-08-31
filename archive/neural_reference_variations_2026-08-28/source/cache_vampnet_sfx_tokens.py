"""Encode a prepared SFX corpus once and cache deterministic LAC tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from run_vampnet_reference_variations import require_safe_gpu
from vampnet_reference_variations import (
    SAMPLE_RATE,
    fix_length,
    prepare_codec_input,
    validate_model_assets,
)


def _load_batch(paths: list[Path], frames: int) -> np.ndarray:
    items = []
    for path in paths:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if int(sample_rate) != SAMPLE_RATE:
            raise ValueError(f"Неожиданная sample rate {sample_rate}: {path}")
        mono = audio.mean(axis=1, dtype=np.float32)
        mono = fix_length(mono, frames)
        items.append(prepare_codec_input(mono))
    return np.stack(items, axis=0)[:, None, :]


def cache_tokens(
    *,
    corpus_dir: Path,
    model_dir: Path,
    output_path: Path,
    batch_size: int,
) -> Path:
    if output_path.exists():
        raise ValueError(f"Token cache уже существует: {output_path}")
    if not 1 <= batch_size <= 8:
        raise ValueError("batch_size должен быть в диапазоне 1..8")
    assets = validate_model_assets(model_dir, required=("codec.pth",))
    gpu = require_safe_gpu()
    manifest_path = corpus_dir / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Не найден corpus manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest.get("items", []))
    if not rows:
        raise ValueError("Corpus manifest не содержит items")
    rows.sort(key=lambda row: (str(row["split"]), str(row["output_file"])))
    paths = [corpus_dir / str(row["output_file"]) for row in rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Не найден prepared WAV: {missing[0]}")

    first_audio, first_rate = sf.read(paths[0], dtype="float32", always_2d=True)
    if int(first_rate) != SAMPLE_RATE:
        raise ValueError("Prepared corpus должен иметь sample rate 44100 Hz")
    frames = int(first_audio.shape[0])

    from lac.model.lac import LAC

    print("[+] Загрузка LAC codec для token cache...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    codec = LAC.load(model_dir / "codec.pth", map_location="cpu")
    codec.eval().requires_grad_(False).to("cuda")
    code_batches: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        audio = _load_batch(batch_paths, frames)
        tensor = torch.from_numpy(audio).to(device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            codes = codec.encode(tensor, SAMPLE_RATE)["codes"]
        code_batches.append(codes.detach().cpu().numpy().astype(np.int16, copy=False))
        if start == 0 or start + batch_size >= len(paths) or start % (20 * batch_size) == 0:
            print(f"    encoded {min(start + len(batch_paths), len(paths))}/{len(paths)}", flush=True)
    all_codes = np.concatenate(code_batches, axis=0)
    if all_codes.shape[0] != len(rows) or all_codes.shape[1] != 14:
        raise RuntimeError(f"Некорректная форма token cache: {all_codes.shape}")

    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "train"],
        dtype=np.int64,
    )
    val_indices = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "val"],
        dtype=np.int64,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        codes=all_codes,
        train_indices=train_indices,
        val_indices=val_indices,
        files=np.asarray([str(row["output_file"]) for row in rows]),
        categories=np.asarray([str(row["category"]) for row in rows]),
    )
    report = {
        "stage": "vampnet_sfx_token_cache",
        "corpus_manifest": str(manifest_path.resolve()),
        "model_assets": assets,
        "gpu": gpu,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
        "normalization": "deterministic -24 dBFS RMS before LAC encode",
        "audio_frames": frames,
        "code_shape": list(all_codes.shape),
        "train_items": int(train_indices.size),
        "validation_items": int(val_indices.size),
        "token_cache": str(output_path.resolve()),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[+] Token cache: {output_path.resolve()}", flush=True)
    print(f"[+] Peak VRAM: {report['peak_vram_mib']:.0f} MiB", flush=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache LAC tokens for SFX LoRA pilot")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/vampnet_models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cache_tokens(
            corpus_dir=args.corpus_dir,
            model_dir=args.model_dir,
            output_path=args.output,
            batch_size=args.batch_size,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[!] Token cache blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
