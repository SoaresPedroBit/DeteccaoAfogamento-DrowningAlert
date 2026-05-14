# Diario de Testes - Deteccao de Afogamento em Piscinas

---

## Teste 01 - Baseline com Dataset v4
**Data:** 2026-04-13

### Dataset
- **Versao:** Roboflow v4
- **Total de imagens:** 2106 (1684 train / 277 valid / 145 test)
- **Numero de classes:** 4
  - piscinas: 4343 instancias (44.4%)
  - crianca_ok: 2538 instancias (26.0%)
  - adulto_ok: 2531 instancias (25.9%)
  - afogamento: 364 instancias (3.7%)
- **Total instancias:** 9776

### Configuracao
- imgsz=640, epochs=200, batch=16, patience=20
- Optimizer: AdamW (lr=0.00125, cos_lr)
- Augmentacao: flipud=0.3, fliplr=0.5, degrees=15, mosaic=1.0, mixup=0.1
- GPU: RTX 4050 6GB
- NMSFree do YOLO26 estava desativado.

### Resultados

| Metrica      | yolo11n | yolo26n |
|--------------|---------|---------|
| Precision    | 0.6715  | 0.6385  |
| Recall       | 0.6330  | 0.6190  |
| mAP@50       | 0.6431  | 0.6099  |
| mAP@50-95    | 0.4084  | 0.4016  |
| Best epoch   | 131     | 130     |
| Tempo treino | 66.7min | 78.6min |

### Opiniao

O primeiro teste serviu como **baseline**. Os resultados sao razoaveis para um dataset pequeno e desbalanceado, mas ha espaco claro para melhoria.

**Pontos positivos:**
- Ambos os modelos convergiram bem, com early stopping ativando por volta da epoch 130-131, o que indica que 200 epochs com patience=20 e uma configuracao adequada.
- O yolo11n superou o yolo26n em todas as metricas, sendo tambem 18% mais rapido no treino. Para este dataset, a arquitetura mais simples do v11n mostrou-se mais eficiente.
- Batch=16 funcionou na RTX 4050 sem OOM, o que e bom para a velocidade de treino.

**Pontos de atencao:**
- O mAP@50 de ~0.64 e o mAP@50-95 de ~0.40 indicam que o modelo detecta razoavelmente, mas a precisao das bounding boxes precisa melhorar.
- O grande desbalanceamento da classe afogamento (apenas 3.7% das instancias, 364 no total) e o principal gargalo. Mesmo com augmentacao agressiva, 364 instancias sao muito poucas para o modelo generalizar bem nesta classe — que e justamente a mais critica.
- O recall de ~0.63 significa que cerca de 37% dos objetos nao estao a ser detectados, o que e preocupante para um sistema de seguranca.

**Proximos passos:**
- Adicionar mais imagens da classe afogamento ao dataset (nova versao no Roboflow).
- Avaliar metricas por classe para confirmar se afogamento e de facto a classe com pior desempenho.
- Considerar tecnicas adicionais: class weights, oversampling, ou focal loss.

---

## Teste 02 - Dataset v5 com mais imagens de afogamento + NMS-free
**Data:** 2026-04-22 (yolo11n, yolo26n) / 2026-04-28 (yolo26s)

### Dataset
- **Versao:** Roboflow v5
- **Total de imagens:** 7057 (6351 train / 423 valid / 283 test)
- **Numero de classes:** 4
  - piscinas: 13155 instancias (40.7%)
  - adulto_ok: 8899 instancias (27.6%)
  - crianca_ok: 8411 instancias (26.0%)
  - afogamento: 1824 instancias (5.6%)
- **Total instancias:** 32289
- **Mudancas vs v4:** Adicionadas imagens de afogamento (364 -> 1824, 5x mais). Augmentacoes Roboflow: Flip Horizontal, Brightness (-25% a +25%), Blur (ate 1.5px), Noise (ate 2%), Crop (0% a 20%). Split: 75/15/10.

### Configuracao
- imgsz=640, epochs=200, patience=20
- batch=16 (yolo11n, yolo26n) / batch=8 (yolo26s, por restricao de VRAM)
- Optimizer: AdamW (lr=0.00125, cos_lr)
- Augmentacao treino: flipud=0.3, fliplr=0.5, degrees=15, mosaic=1.0, mixup=0.1
- GPU: RTX 4050 6GB
- **yolo26n e yolo26s com end2end=True (NMS-free) ativado**
- yolo26s: ~9.47M parametros (vs ~2.6M do yolo26n), variante "small" da familia YOLO26

### Resultados

| Metrica      | yolo11n | yolo26n (end2end) | yolo26s (end2end) |
|--------------|---------|-------------------|-------------------|
| Precision    | 0.7754  | 0.7666            | 0.7794            |
| Recall       | 0.7155  | 0.7216            | 0.7152            |
| mAP@50       | 0.7375  | 0.7235            | 0.7284            |
| mAP@50-95    | 0.4929  | 0.4936            | 0.5043            |
| Best epoch   | 97      | 121               | 99                |
| Tempo treino | ~1066min| 244.9min          | 367.8min          |

### Metricas por classe (split de teste)

**yolo11n:**

| Classe     | Precision | Recall | mAP@50 | mAP@50-95 |
|------------|-----------|--------|--------|-----------|
| piscinas   | 0.7892    | 0.7086 | 0.7943 | 0.6271    |
| crianca_ok | 0.7111    | 0.6625 | 0.6554 | 0.3993    |
| adulto_ok  | 0.7069    | 0.6249 | 0.6472 | 0.3719    |
| afogamento | 0.6582    | 0.5890 | 0.6401 | 0.3706    |

**yolo26n (end2end):**

| Classe     | Precision | Recall | mAP@50 | mAP@50-95 |
|------------|-----------|--------|--------|-----------|
| piscinas   | 0.7757    | 0.7176 | 0.7925 | 0.6476    |
| crianca_ok | 0.6952    | 0.6750 | 0.6616 | 0.4000    |
| adulto_ok  | 0.7108    | 0.6727 | 0.6616 | 0.4122    |
| afogamento | 0.7090    | 0.6592 | 0.6228 | 0.3623    |

**yolo26s (end2end):**

| Classe     | Precision | Recall | mAP@50 | mAP@50-95 |
|------------|-----------|--------|--------|-----------|
| piscinas   | 0.7586    | 0.6923 | 0.7695 | 0.6444    |
| crianca_ok | 0.6981    | 0.6583 | 0.6627 | 0.4344    |
| adulto_ok  | 0.7545    | 0.6460 | 0.6370 | 0.4038    |
| afogamento | 0.6858    | 0.6421 | 0.5943 | 0.3480    |

**Classe afogamento - comparativo direto:**

| Metrica   | yolo11n | yolo26n | yolo26s | Melhor  |
|-----------|---------|---------|---------|---------|
| Precision | 0.6582  | 0.7090  | 0.6858  | yolo26n (+5.1pp vs 11n)  |
| Recall    | 0.5890  | 0.6592  | 0.6421  | yolo26n (+7.0pp vs 11n)  |
| mAP@50    | 0.6401  | 0.6228  | 0.5943  | yolo11n (+1.7pp vs 26n)  |
| mAP@50-95 | 0.3706  | 0.3623  | 0.3480  | yolo11n (+0.8pp vs 26n)  |

O yolo26n continua a ser a melhor opcao para a classe afogamento (Recall e Precision mais altos). O yolo26s, apesar de ter mais parametros, nao superou o yolo26n na deteccao de afogamento — provavelmente por overfitting com o dataset ainda limitado nesta classe (1824 instancias). Para um sistema de seguranca onde falsos negativos sao criticos, o yolo26n com end2end continua a melhor escolha.

### Evolucao vs Teste 01

| Metrica    | T01 yolo11n | T02 yolo11n | Melhoria |
|------------|------------|------------|----------|
| Precision  | 0.6715     | 0.7754     | +10.4pp  |
| Recall     | 0.6330     | 0.7155     | +8.3pp   |
| mAP@50     | 0.6431     | 0.7375     | +9.4pp   |
| mAP@50-95  | 0.4084     | 0.4929     | +8.5pp   |

### Opiniao

Melhoria significativa em relacao ao Teste 01. O aumento do dataset (2106 -> 7057 imagens) e o foco no reequilibrio da classe afogamento (364 -> 1824 instancias) foram os fatores decisivos.

**Pontos positivos:**
- Todas as metricas subiram ~9pp em relacao ao Teste 01. O dataset maior e mais equilibrado fez a diferenca.
- yolo11n, yolo26n e yolo26s ficaram num intervalo muito estreito. yolo11n lidera em mAP@50, yolo26n em Recall geral, e yolo26s em Precision e mAP@50-95.
- O Recall do yolo26n (0.7216) e ligeiramente superior, o que e relevante para deteccao de afogamento — menos falsos negativos significa menos afogamentos nao detectados.
- yolo26s alcancou o melhor mAP@50-95 (0.5043), confirmando que a variante "small" produz bounding boxes mais precisas. Trade-off: nao trouxe ganho na classe afogamento especificamente.

**Pontos de atencao:**
- A classe afogamento ainda representa apenas 5.6% das instancias. Idealmente deveria chegar a 10-15%.
- O tempo do yolo11n (~1066min) inclui a interrupcao por fecho do notebook. O tempo real seria proximo de ~250min.
- O yolo26s levou 367.8min com batch=8 (limitado pela VRAM de 6GB). Em GPU maior, com batch=16, tempo cairia ~50%.
- mAP@50-95 de ~0.50 mostra que a precisao das bounding boxes ainda pode melhorar.
- yolo26s nao superou yolo26n na classe afogamento (Recall -1.7pp, mAP@50 -2.8pp) — modelo maior nao compensou a falta de dados desta classe.

**Proximos passos:**
- Continuar a aumentar imagens de afogamento no dataset (meta: 10-15% do total).
- Considerar testar com imgsz=1280 se a VRAM permitir (melhor deteccao de objetos pequenos).
- Avaliar yolo26n como modelo principal para deploy dado o melhor Recall na classe afogamento.
- Reavaliar yolo26s com dataset v6 — modelo maior pode brilhar quando a classe afogamento estiver melhor representada.

---

## Teste 03 - Dataset v6 + hiperparametros otimizados para Recall
**Data:** pendente
**Objetivo:** Atingir Recall >= 0.80 e Precision >= 0.80 na classe afogamento.

### Dataset
- **Versao:** Roboflow v6 (pendente)
- **Foco:** Continuar aumentando imagens da classe afogamento (meta: 4000-5000 instancias, 12-15% do total)

### Mudancas em relacao ao Teste 02

**Hiperparametros ajustados:**

| Parametro    | Teste 02 | Teste 03 | Motivo |
|-------------|----------|----------|--------|
| epochs      | 200      | 300      | Mais tempo para convergir com dataset maior |
| patience    | 20       | 30       | Evita early stopping prematuro |
| cls         | 0.5      | 1.0      | Mais peso na classificacao — ajuda a distinguir afogamento |
| close_mosaic| 10       | 20       | Mais epochs finais sem mosaic para refinar deteccoes |

**Mantidos do Teste 02:**
- yolo26n com end2end=True (NMS-free) — melhor Recall na classe afogamento
- Augmentacoes: flipud=0.3, fliplr=0.5, degrees=15, mosaic=1.0, mixup=0.1
- imgsz=640, batch=16, cos_lr=True

**Deploy:**
- Alvo: servidores cloud com GPU (T4/A10/L4)
- yolo26n nano mantido (latencia ~9ms/frame, suficiente para alerta em tempo real)
- Conf threshold baixo (~0.15-0.20) na inferencia para maximizar Recall

### Resultados
Pendente — aguardando novo dataset.

---
