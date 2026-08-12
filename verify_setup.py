import sys
import torch
import librosa
import torchaudio
import diffusers

def run_diagnostics():
    print("=" * 60)
    print(" ДИАГНОСТИКА ОКРУЖЕНИЯ MINICONDA (sfx_gen) ")
    print("=" * 60)
    
    print(f"[+] Версия Python: {sys.version.split()[0]}")
    print(f"[+] Версия PyTorch: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"[+] Поддержка CUDA доступна: {cuda_available}")
    
    if cuda_available:
        print(f"[+] Название видеокарты: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[+] Доступный объем VRAM: {vram_gb:.2f} GB")
        print(f"[+] Версия CUDA в PyTorch: {torch.version.cuda}")
    else:
        print("[!] ВНИМАНИЕ: CUDA не обнаружена.")

    print(f"[+] Версия Librosa: {librosa.__version__}")
    print(f"[+] Версия TorchAudio: {torchaudio.__version__}")
    print(f"[+] Версия Diffusers: {diffusers.__version__}")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()