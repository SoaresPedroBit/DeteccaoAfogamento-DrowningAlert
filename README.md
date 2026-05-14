# Deteccao de Afogamento em Piscinas com YOLO

Trabalho de Conclusao de Curso (TCC) que compara modelos YOLO (`yolo11n`, `yolo26n` e `yolo26s`) na tarefa de deteccao de afogamento em piscinas a partir de imagens. O dataset e mantido no Roboflow e contem 4 classes: `piscinas`, `adulto_ok`, `crianca_ok` e `afogamento`.

## Sumario
- [Visao Geral](#visao-geral)
- [Dataset](#dataset)
- [Ambiente](#ambiente)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Executar](#como-executar)
- [Configuracao de Treino](#configuracao-de-treino)
- [Resultados](#resultados)

## Visao Geral
O objetivo e avaliar o desempenho de variantes do YOLO na deteccao de afogamento, classe naturalmente sub-representada em datasets de piscinas. O pipeline cobre desde a validacao do ambiente CUDA ate o treino, validacao e comparativo dos modelos.

## Dataset
- **Origem:** Roboflow - workspace `pedros-workspace-jewao`, projeto `deteccao_afogamento`.
- **Versao atual:** v5 - 7057 imagens (6351 train / 423 valid / 283 test).
- **Formato:** `yolov11` (compativel com todos os modelos usados).
- **Classes e instancias (v5):** piscinas=13155, adulto_ok=8899, crianca_ok=8411, afogamento=1824.
- **Augmentacoes do Roboflow:** flip horizontal, brilho (-25% a +25%), blur (ate 1.5px), ruido (ate 2%), crop (0% a 20%).

> O diretorio `datasets/` esta no `.gitignore`. Para obter o dataset, configure `ROBOFLOW_API_KEY` e rode `scripts/02_baixar_dataset.py`.

## Ambiente
- **SO:** Windows 11
- **GPU:** NVIDIA RTX 4050 Laptop (6 GB VRAM)
- **Python:** 3.11 via Miniconda (ambiente `tcc`)
- **PyTorch:** 2.11.0 + CUDA 12.6
- **Ultralytics:** 8.4.36
- **Roboflow:** 1.3.1

## Estrutura do Projeto
```
TCC/
  DIARIO_TESTES.md
  README.md
  .gitignore
  scripts/
    01_verificar_ambiente.py   # Valida CUDA, PyTorch e dependencias
    02_baixar_dataset.py       # Baixa o dataset do Roboflow
    03_treinar.py              # Treina yolo11n e/ou yolo26n
    03b_treinar_yolo26s.py     # Treina yolo26s (variante "small")
    04_comparar_modelos.py     # Gera comparativo de metricas
  datasets/                    # (ignorado) dataset baixado do Roboflow
  runs/                        # (parcial) resultados de treino
    yolo11n/                   # plots, csv, args - pesos ignorados
    yolo26n/
    yolo26s/
    comparativo_modelos.png
```

## Como Executar
```bash
conda activate tcc

# 1. Validar ambiente (CUDA, libs, GPU)
python scripts/01_verificar_ambiente.py

# 2. Baixar dataset (requer chave do Roboflow)
export ROBOFLOW_API_KEY="<sua_chave>"   # bash
# $env:ROBOFLOW_API_KEY = "<sua_chave>" # PowerShell
python scripts/02_baixar_dataset.py

# 3. Treinar
python scripts/03_treinar.py                   # ambos (yolo11n + yolo26n)
python scripts/03_treinar.py --modelo yolo11n  # apenas um
python scripts/03b_treinar_yolo26s.py          # variante small

# 4. Comparar metricas
python scripts/04_comparar_modelos.py
```

## Configuracao de Treino
- `imgsz=640`, `epochs=200-300`, `batch=16` (ou 8 em fallback de OOM)
- Optimizer `AdamW` (auto), `cos_lr=True`, `patience=20-30`
- Augmentacao: `fliplr=0.5`, `flipud=0.3`, `degrees=15`, `mosaic=1.0`, `mixup=0.1`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `translate=0.1`, `scale=0.5`
- `yolo26n` e `yolo26s` treinados com `end2end=True` (NMS-free)
- Fallback automatico de batch `16 -> 8` em caso de OOM

## Resultados

### Treino 1 - Dataset v4 (yolo26n sem end2end)
| Metrica      | yolo11n | yolo26n |
|--------------|---------|---------|
| Precision    | 0.6715  | 0.6385  |
| Recall       | 0.6330  | 0.6190  |
| mAP@50       | 0.6431  | 0.6099  |
| mAP@50-95    | 0.4084  | 0.4016  |
| Best epoch   | 131     | 130     |
| Tempo treino | 66.7min | 78.6min |

### Treino 2 - Dataset v5 (yolo26n e yolo26s com end2end)
| Metrica      | yolo11n  | yolo26n (end2end) | yolo26s (end2end) |
|--------------|----------|-------------------|-------------------|
| Precision    | 0.7754   | 0.7666            | **0.7794**        |
| Recall       | 0.7155   | **0.7216**        | 0.7152            |
| mAP@50       | **0.7375** | 0.7235          | 0.7284            |
| mAP@50-95    | 0.4929   | 0.4936            | **0.5043**        |
| Best epoch   | 97       | 121               | 99                |
| Tempo treino | ~1066min | 244.9min          | 367.8min          |

**Conclusoes:**
- Os tres modelos ficam num intervalo estreito de desempenho.
- `yolo11n` lidera em mAP@50; `yolo26n` em Recall; `yolo26s` em Precision e mAP@50-95.
- `yolo26s` troca um pouco de Recall (-0.6pp vs `yolo26n`) por +1.07pp em mAP@50-95 - melhor escolha quando a qualidade do bounding box e prioridade.
- A migracao v4 -> v5 do dataset trouxe ganhos de ~9pp em todas as metricas; balanceamento e volume foram os fatores decisivos.

---
Projeto academico - TCC.
