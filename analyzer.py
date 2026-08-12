"""DSP-утилиты для работы с референсной и сгенерированной огибающей."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_audio(file_path: str | Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """Загрузить WAV, привести его к mono и указанной частоте дискретизации."""
    import librosa

    samples, sample_rate = librosa.load(str(file_path), sr=target_sr, mono=True)
    return torch.tensor(samples, dtype=torch.float32), sample_rate


def normalize_01(values: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Нормировать одномерный тензор в [0, 1] без деления на ноль."""
    minimum = values.min()
    maximum = values.max()
    span = maximum - minimum
    if float(span.detach().abs().cpu()) <= eps:
        return torch.zeros_like(values)
    return (values - minimum) / span


def extract_rms_envelope(
    samples: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """Извлечь нормированную RMS-огибающую на PyTorch.

    Функция принимает моно-сигнал формы ``[samples]``. Для очень коротких
    файлов дополняет его нулями до одного окна, поэтому возвращает хотя бы
    одну точку вместо ошибки ``unfold``.
    """
    if samples.ndim != 1:
        raise ValueError(f"Ожидался моно-тензор [samples], получено {tuple(samples.shape)}")
    if samples.numel() == 0:
        raise ValueError("Нельзя извлечь огибающую из пустого аудиосигнала")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length и hop_length должны быть положительными")

    if samples.numel() < frame_length:
        samples = F.pad(samples, (0, frame_length - samples.numel()))

    frames = samples.unfold(0, frame_length, hop_length)
    rms = torch.sqrt(torch.mean(frames.square(), dim=-1) + 1e-8)
    return normalize_01(rms)


def plot_and_save(
    samples: torch.Tensor,
    rms_normalized: torch.Tensor,
    sample_rate: int,
    output_image: str | Path = "envelope_result.png",
) -> None:
    """Сохранить график исходного сигнала и его RMS-огибающей."""
    import matplotlib.pyplot as plt

    samples_cpu = samples.detach().cpu().numpy()
    envelope_cpu = rms_normalized.detach().cpu().numpy()
    time_audio = np.linspace(0, len(samples_cpu) / sample_rate, num=len(samples_cpu))
    time_rms = np.linspace(0, len(samples_cpu) / sample_rate, num=len(envelope_cpu))

    plt.figure(figsize=(10, 4))
    plt.plot(time_audio, samples_cpu, label="Исходный сигнал", color="gray", alpha=0.5)
    plt.plot(time_rms, envelope_cpu, label="RMS-огибающая", color="red", linewidth=2)
    plt.title("Извлечение целевой огибающей E_target")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда / нормированная энергия")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"[+] График анализа сохранен: {output_image}")


if __name__ == "__main__":
    import soundfile as sf

    input_file = Path("test.wav")
    if not input_file.exists():
        print(f"[!] Файл '{input_file}' не найден. Создается затухающий тестовый сигнал...")
        sample_rate = 16000
        time_axis = np.linspace(0, 1.5, int(sample_rate * 1.5), endpoint=False)
        signal = np.sin(2 * np.pi * 440 * time_axis) * np.exp(-4 * time_axis)
        sf.write(input_file, signal, sample_rate)

    audio_tensor, sample_rate = load_audio(input_file)
    envelope_tensor = extract_rms_envelope(audio_tensor)
    print(
        f"[+] Аудио: {len(audio_tensor) / sample_rate:.2f} с | "
        f"частота: {sample_rate} Гц | точек огибающей: {len(envelope_tensor)}"
    )
    plot_and_save(audio_tensor, envelope_tensor, sample_rate)
