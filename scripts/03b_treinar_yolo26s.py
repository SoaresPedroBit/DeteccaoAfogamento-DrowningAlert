"""
Teste comparativo: yolo26s com parametros do Teste 02.
Objetivo: avaliar se o modelo small traz ganho vs nano na classe afogamento.

Uso:
    python 03b_treinar_yolo26s.py
    python 03b_treinar_yolo26s.py --batch 8
"""
import argparse
import os
import sys
import time

import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_YAML = os.path.join(PROJECT_ROOT, "datasets", "deteccao_afogamento", "data.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Parametros do Teste 02 (baseline para comparacao justa)
TRAIN_PARAMS = {
    "imgsz": 640,
    "epochs": 200,
    "patience": 20,
    "batch": 16,
    "workers": 4,
    "optimizer": "auto",
    "cos_lr": True,
    "save": True,
    "save_period": 25,
    "plots": True,
    "val": True,
    "exist_ok": True,
}

AUGMENTATION_PARAMS = {
    "flipud": 0.3,
    "fliplr": 0.5,
    "degrees": 15.0,
    "mosaic": 1.0,
    "mixup": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "translate": 0.1,
    "scale": 0.5,
}


def detectar_batch():
    if not torch.cuda.is_available():
        return 8
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # yolo26s e maior que nano, batch=8 e mais seguro com 6GB
    if vram_gb >= 8:
        return 16
    else:
        print(f"[INFO] VRAM={vram_gb:.1f}GB - usando batch=8 para yolo26s")
        return 8


def main():
    parser = argparse.ArgumentParser(description="Treino yolo26s para comparativo")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not os.path.isfile(DATASET_YAML):
        print(f"[ERRO] Dataset nao encontrado: {DATASET_YAML}")
        print("Execute primeiro: python 02_baixar_dataset.py")
        sys.exit(1)

    device = args.device if args.device else ("0" if torch.cuda.is_available() else "cpu")
    batch = args.batch if args.batch else detectar_batch()

    print(f"\n{'=' * 60}")
    print(f"TREINANDO: yolo26s (comparativo com parametros do Teste 02)")
    print(f"Weights: yolo26s.pt | Batch: {batch} | Device: {device}")
    print(f"end2end=True (NMS-free)")
    print(f"{'=' * 60}\n")

    model = YOLO("yolo26s.pt")

    params = {
        **TRAIN_PARAMS,
        **AUGMENTATION_PARAMS,
        "end2end": True,
        "data": DATASET_YAML,
        "project": RUNS_DIR,
        "name": "yolo26s",
        "batch": batch,
        "device": device,
    }

    inicio = time.time()
    try:
        results = model.train(**params)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"\n[AVISO] OOM com batch={batch}. Reduzindo para batch={batch // 2}...")
            torch.cuda.empty_cache()
            params["batch"] = batch // 2
            results = model.train(**params)
        else:
            raise

    duracao = time.time() - inicio
    print(f"\n[OK] yolo26s treinado em {duracao / 60:.1f} minutos")
    print(f"Resultados salvos em: {RUNS_DIR}/yolo26s")

    return results


if __name__ == "__main__":
    main()
