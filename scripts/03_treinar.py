"""
Script de treino YOLO para deteccao de afogamento.
Treina yolo11n e yolo26n com augmentacao para classe desbalanceada.

Uso:
    python 03_treinar.py                    # treina ambos os modelos
    python 03_treinar.py --modelo yolo11n   # treina apenas yolo11n
    python 03_treinar.py --modelo yolo26n   # treina apenas yolo26n
    python 03_treinar.py --batch 8          # forca batch=8
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

MODELOS = {
    "yolo11n": {"weights": "yolo11n.pt", "extra": {}},
    "yolo26n": {"weights": "yolo26n.pt", "extra": {"end2end": True}},
}

# Hiperparametros de treino
TRAIN_PARAMS = {
    "imgsz": 640,
    "epochs": 300,
    "patience": 30,
    "batch": 16,
    "workers": 4,
    "optimizer": "auto",
    "cos_lr": True,
    "cls": 2.0,
    "close_mosaic": 20,
    "save": True,
    "save_period": 25,
    "plots": True,
    "val": True,
    "exist_ok": True,
}

# Augmentacao para compensar desequilibrio da classe afogamento
AUGMENTATION_PARAMS = {
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
    """Detecta batch ideal baseado na VRAM disponivel."""
    if not torch.cuda.is_available():
        print("[INFO] Sem GPU, usando batch=8 em CPU")
        return 8

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= 8:
        return 16
    elif vram_gb >= 5:
        print(f"[INFO] VRAM={vram_gb:.1f}GB - tentando batch=16, fallback para 8 se OOM")
        return 16
    else:
        print(f"[INFO] VRAM={vram_gb:.1f}GB - usando batch=8")
        return 8


def treinar_modelo(modelo_nome, modelo_config, batch, device):
    """Treina um modelo YOLO e retorna o caminho dos resultados."""
    modelo_weights = modelo_config["weights"]
    extra_params = modelo_config.get("extra", {})

    print(f"\n{'=' * 60}")
    print(f"TREINANDO: {modelo_nome}")
    print(f"Weights: {modelo_weights} | Batch: {batch} | Device: {device}")
    if extra_params:
        print(f"Params extras: {extra_params}")
    print(f"{'=' * 60}\n")

    model = YOLO(modelo_weights)

    params = {
        **TRAIN_PARAMS,
        **AUGMENTATION_PARAMS,
        **extra_params,
        "data": DATASET_YAML,
        "project": RUNS_DIR,
        "name": modelo_nome,
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
    print(f"\n[OK] {modelo_nome} treinado em {duracao / 60:.1f} minutos")

    return results


def main():
    parser = argparse.ArgumentParser(description="Treino YOLO para deteccao de afogamento")
    parser.add_argument("--modelo", choices=["yolo11n", "yolo26n"], default=None,
                        help="Modelo especifico para treinar (default: ambos)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Tamanho do batch (default: auto-detectar)")
    parser.add_argument("--device", default=None,
                        help="Device: 0 para GPU, cpu para CPU (default: auto)")
    args = parser.parse_args()

    # Validar dataset
    if not os.path.isfile(DATASET_YAML):
        print(f"[ERRO] Dataset nao encontrado: {DATASET_YAML}")
        print("Execute primeiro: python 02_baixar_dataset.py")
        sys.exit(1)

    # Configurar device
    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # Configurar batch
    batch = args.batch if args.batch else detectar_batch()

    # Selecionar modelos
    modelos_para_treinar = {}
    if args.modelo:
        modelos_para_treinar[args.modelo] = MODELOS[args.modelo]
    else:
        modelos_para_treinar = MODELOS

    # Treinar
    resultados = {}
    for nome, config in modelos_para_treinar.items():
        resultados[nome] = treinar_modelo(nome, config, batch, device)

    print(f"\n{'=' * 60}")
    print("TREINO CONCLUIDO")
    print(f"Resultados salvos em: {RUNS_DIR}")
    print(f"{'=' * 60}")

    return resultados


if __name__ == "__main__":
    main()
