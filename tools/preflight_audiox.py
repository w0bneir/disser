"""Read-only system and artifact gate for AudioX experiments.

The script deliberately does not import torch or construct the model.  It is
safe to run before every AudioX process and fails closed when the checkpoint,
configuration, RAM, disk, or GPU headroom do not meet the audited limits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_CHECKPOINT_BYTES = 5_957_258_603
MINIMUM_TOTAL_VRAM_MIB = 12_000
MINIMUM_FREE_VRAM_MIB = 10_000
MINIMUM_TOTAL_RAM_GIB = 28.0
MINIMUM_FREE_RAM_GIB = 18.0
MINIMUM_FREE_DISK_GIB = 16.0


class PreflightError(RuntimeError):
    """A failed, actionable AudioX gate."""


def configure_windows_console() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver: str
    total_mib: int
    free_mib: int


def parse_nvidia_smi_line(line: str) -> GpuInfo:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        raise PreflightError(f"Не удалось разобрать вывод nvidia-smi: {line!r}")
    try:
        total_mib = int(parts[2])
        free_mib = int(parts[3])
    except ValueError as error:
        raise PreflightError(f"Некорректный объём VRAM в выводе nvidia-smi: {line!r}") from error
    return GpuInfo(parts[0], parts[1], total_mib, free_mib)


def query_gpu() -> GpuInfo:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise PreflightError("nvidia-smi не найден или NVIDIA GPU не отвечает") from error
    first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line:
        raise PreflightError("nvidia-smi вернул пустой ответ")
    return parse_nvidia_smi_line(first_line)


def validate_config(config: dict[str, Any]) -> None:
    try:
        model = config["model"]
        diffusion = model["diffusion"]["config"]
        pretransform = model["pretransform"]["config"]
        conditioner_ids = {
            item["id"] for item in model["conditioning"]["configs"]
        }
    except (KeyError, TypeError) as error:
        raise PreflightError("AudioX config имеет неизвестную структуру") from error

    expected = {
        "model_type": (config.get("model_type"), "diffusion_cond"),
        "sample_rate": (config.get("sample_rate"), 44_100),
        "sample_size": (config.get("sample_size"), 485_100),
        "audio_channels": (config.get("audio_channels"), 2),
        "io_channels": (diffusion.get("io_channels"), 64),
        "embed_dim": (diffusion.get("embed_dim"), 1_536),
        "depth": (diffusion.get("depth"), 24),
        "num_heads": (diffusion.get("num_heads"), 24),
        "latent_dim": (pretransform.get("latent_dim"), 64),
        "downsampling_ratio": (pretransform.get("downsampling_ratio"), 2_048),
    }
    mismatches = [
        f"{name}={actual!r}, ожидалось {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if "audio_prompt" not in conditioner_ids:
        mismatches.append("нет обязательного conditioner audio_prompt")
    if mismatches:
        raise PreflightError("Непроверенная конфигурация AudioX: " + "; ".join(mismatches))


def find_cached_checkpoint(cache_root: Path) -> Path | None:
    snapshots = (
        cache_root
        / "hub"
        / "models--HKUSTAudio--AudioX"
        / "snapshots"
    )
    if not snapshots.exists():
        return None
    candidates = sorted(snapshots.glob("*/model.ckpt"))
    return candidates[-1] if candidates else None


def validate_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise PreflightError(f"Checkpoint AudioX не найден: {path}")
    size = path.stat().st_size
    if size != EXPECTED_CHECKPOINT_BYTES:
        raise PreflightError(
            f"Checkpoint AudioX неполный или другой версии: {size} байт; "
            f"ожидалось {EXPECTED_CHECKPOINT_BYTES}"
        )


def memory_gib() -> tuple[float, float]:
    try:
        import psutil
    except ImportError as error:
        raise PreflightError("Для RAM gate требуется пакет psutil") from error
    memory = psutil.virtual_memory()
    return memory.total / 2**30, memory.available / 2**30


def run_preflight(arguments: argparse.Namespace) -> Path:
    project_root = arguments.project_root.resolve()
    config_path = arguments.config.resolve()
    if not config_path.is_file():
        raise PreflightError(f"AudioX config не найден: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        validate_config(json.load(stream))

    checkpoint = arguments.checkpoint
    if checkpoint is None:
        checkpoint = find_cached_checkpoint(arguments.cache_root.resolve())
    if checkpoint is None:
        raise PreflightError("Checkpoint AudioX отсутствует в локальном cache")
    checkpoint = checkpoint.resolve()
    validate_checkpoint(checkpoint)

    if importlib.util.find_spec("audiox") is None:
        raise PreflightError(
            "Пакет audiox не найден: запускайте gate через artifacts\\audiox_env"
        )

    total_ram, free_ram = memory_gib()
    if total_ram < MINIMUM_TOTAL_RAM_GIB or free_ram < MINIMUM_FREE_RAM_GIB:
        raise PreflightError(
            "Недостаточно RAM: "
            f"{total_ram:.1f} GiB всего, {free_ram:.1f} GiB доступно; "
            f"нужно >= {MINIMUM_TOTAL_RAM_GIB:.0f}/{MINIMUM_FREE_RAM_GIB:.0f} GiB"
        )

    free_disk = shutil.disk_usage(project_root).free / 2**30
    if free_disk < MINIMUM_FREE_DISK_GIB:
        raise PreflightError(
            f"Недостаточно диска: {free_disk:.1f} GiB; нужно >= {MINIMUM_FREE_DISK_GIB:.0f} GiB"
        )

    gpu = query_gpu()
    if (
        gpu.total_mib < MINIMUM_TOTAL_VRAM_MIB
        or gpu.free_mib < MINIMUM_FREE_VRAM_MIB
    ):
        raise PreflightError(
            "Недостаточно VRAM: "
            f"{gpu.total_mib} MiB всего, {gpu.free_mib} MiB свободно; "
            f"нужно >= {MINIMUM_TOTAL_VRAM_MIB}/{MINIMUM_FREE_VRAM_MIB} MiB"
        )

    print(f"[+] Config: {config_path}")
    print(f"[+] Checkpoint: {checkpoint} ({checkpoint.stat().st_size / 2**30:.2f} GiB)")
    print(f"[+] RAM: {total_ram:.1f} GiB всего, {free_ram:.1f} GiB доступно")
    print(f"[+] Disk: {free_disk:.1f} GiB свободно")
    print(
        f"[+] GPU: {gpu.name}; driver {gpu.driver}; "
        f"VRAM {gpu.total_mib} MiB всего, {gpu.free_mib} MiB свободно"
    )
    print("[+] AudioX preflight: OK; модель не загружалась")
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "artifacts" / "AudioX_source" / "config.json",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=project_root / "artifacts" / "audiox_hf_cache",
    )
    return parser


def main() -> int:
    configure_windows_console()
    try:
        run_preflight(build_parser().parse_args())
    except PreflightError as error:
        print(f"[!] AudioX preflight заблокирован: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
