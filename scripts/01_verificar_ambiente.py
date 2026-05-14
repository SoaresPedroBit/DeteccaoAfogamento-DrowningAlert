"""
Verificacao do ambiente antes do treino.
Valida: Python, PyTorch, CUDA/GPU, VRAM, ultralytics, roboflow.
"""
import sys


def verificar_python():
    versao = sys.version
    print(f"[OK] Python: {versao}")
    assert sys.version_info >= (3, 10), "Python >= 3.10 necessario"


def verificar_pytorch():
    import torch
    print(f"[OK] PyTorch: {torch.__version__}")
    return torch


def verificar_gpu(torch):
    if not torch.cuda.is_available():
        print("[AVISO] CUDA nao disponivel - treino sera em CPU (muito lento)")
        return None, 0

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[OK] GPU: {gpu_name}")
    print(f"[OK] VRAM: {vram_gb:.1f} GB")
    return gpu_name, vram_gb


def verificar_ultralytics():
    import ultralytics
    print(f"[OK] Ultralytics: {ultralytics.__version__}")


def verificar_roboflow():
    import roboflow
    print(f"[OK] Roboflow: {roboflow.__version__}")


def recomendar_batch(vram_gb):
    if vram_gb >= 8:
        batch = 16
    elif vram_gb >= 5:
        batch = 16
        print("[INFO] VRAM no limite para batch=16. Se houver OOM, usar batch=8.")
    elif vram_gb > 0:
        batch = 8
        print(f"[INFO] VRAM={vram_gb:.1f}GB insuficiente para batch=16. Usando batch=8.")
    else:
        batch = 8
        print("[INFO] Sem GPU. batch=8 em CPU.")
    print(f"[OK] Batch recomendado: {batch}")
    return batch


def main():
    print("=" * 50)
    print("VERIFICACAO DO AMBIENTE DE TREINO")
    print("=" * 50)

    verificar_python()
    torch = verificar_pytorch()
    gpu_name, vram_gb = verificar_gpu(torch)
    verificar_ultralytics()
    verificar_roboflow()
    batch = recomendar_batch(vram_gb)

    print("=" * 50)
    print("Ambiente pronto para treino." if gpu_name else "Ambiente pronto (CPU only).")
    print("=" * 50)
    return batch


if __name__ == "__main__":
    main()
