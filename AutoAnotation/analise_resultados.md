# Analise dos Resultados - Auto-anotacao YOLO26N

Relatorio da inferencia automatica do modelo `yolo26n/best.pt` (dataset v5) sobre a playlist [Wavepool Lifeguard Rescue Videos](https://www.youtube.com/playlist?list=PLgqwWmjSsNRzDOAmtyRc9K3fO-VMpE07C).

## Parametros da execucao
- **Modelo:** `runs/yolo26n/weights/best.pt`
- **Playlist:** 73 videos (Wavepool Lifeguard Rescue)
- **Extracao:** 1 frame a cada 3 segundos via OpenCV (imgsz max 640, mantem proporcao)
- **Confianca minima:** 0.25
- **Data:** 2026-05-14

## Numeros gerais
- **Videos processados:** 73
- **Frames extraidos:** 1800
- **Frames com deteccao:** 1796 (99.8% de cobertura)
- **Frames sem deteccao:** 4
- **Total de bounding boxes:** 23.962

### Distribuicao de deteccoes por classe

| Classe | Deteccoes | % do total | Frames com a classe | % dos frames |
|--------|----------:|-----------:|--------------------:|-------------:|
| 0 piscinas    | 16.519 | 68.9% | 1.739 | 96.6% |
| 1 adulto_ok   |     42 |  0.2% |    36 |  2.0% |
| 2 crianca_ok  |  5.422 | 22.6% | 1.203 | 66.8% |
| 3 afogamento  |  1.979 |  8.3% | 1.783 | 99.1% |

### Confianca media (por frame)
- Media: **0.496**
- Mediana: **0.485**
- Min: 0.354
- Max: 0.975

### Composicao dos frames
| Tipo | Frames | % |
|------|-------:|--:|
| multiclasse | 1.776 | 98.7% |
| somente afogamento | 13 | 0.7% |
| somente piscinas | 7 | 0.4% |
| sem deteccao | 4 | 0.2% |

### Area media de bounding box por classe (% da imagem)
| Classe | Media | Mediana | n |
|--------|------:|--------:|--:|
| 0 piscinas    |  0.54% |  0.24% | 16.519 |
| 1 adulto_ok   |  2.30% |  1.67% |     42 |
| 2 crianca_ok  |  0.36% |  0.09% |  5.422 |
| 3 afogamento  | **78.06%** | **81.43%** | 1.979 |

## Achados sobre o comportamento do modelo

### 1. Classe `afogamento` esta sendo predita como uma "regiao da cena" e nao como pessoa
- Bbox media cobre **78%** da imagem (mediana 81%).
- Comparado com as demais classes (todas com bboxes <2.5% em media), eh um padrao destoante.
- Hipotese: no dataset Roboflow v5, a classe afogamento foi anotada com bboxes englobando a cena inteira do afogamento (nao a pessoa especifica). O detector aprendeu essa correlacao.
- **Impacto:** reduz o valor pratico para apontar ONDE esta o afogamento — funciona mais como um classificador binario "cena tem afogamento sim/nao" disfarcado de detector.

### 2. Classe `adulto_ok` quase nao e detectada
- 42 deteccoes em 1800 frames (0.2% do total).
- Ratio crianca_ok / adulto_ok = **129x**.
- Hipoteses combinadas:
  - Bias do dataset de treino (poucos exemplos rotulados como adulto).
  - Playlist tem maioria de jovens/adolescentes ambiguos.
  - Confusao real do modelo entre adulto e crianca em corpos parcialmente submersos.

### 3. Classe `piscinas` proliferada
- 9.2 deteccoes por frame em media (16519 / 1796).
- Bboxes pequenas (mediana 0.24% da imagem).
- Indica que o modelo segmenta a piscina em varias regioes pequenas em vez de uma bbox unica.

### 4. Cobertura mecanica excelente
- 99.8% dos frames tiveram pelo menos uma deteccao.
- 98.7% sao multiclasse (afogamento + piscina + pessoas juntos), o que faz sentido para o conteudo da playlist.

### 5. Confianca proxima do limiar
- Mediana 0.485 com limiar 0.25 sugere muitas deteccoes "no limite".
- Aumentar `conf` para 0.35-0.40 reduziria ruido mas pode descartar afogamentos sutis.

## Top frames para revisao manual

### Mais deteccoes (candidatos a confusao)
1. 34 dets, conf=0.531 - `068_..._frame_0001`
2. 33 dets, conf=0.405 - `028_..._frame_0022`
3. 32 dets, conf=0.406 - `019_..._frame_0023`
4. 32 dets, conf=0.453 - `019_..._frame_0025`
5. 32 dets, conf=0.476 - `028_..._frame_0013`

### Menor confianca (candidatos a falso positivo)
1. conf=0.354, 14 dets - `067_..._frame_0036`
2. conf=0.358, 21 dets - `058_..._frame_0007`
3. conf=0.362, 10 dets - `061_..._frame_0001`
4. conf=0.365, 16 dets - `054_..._frame_0017`
5. conf=0.372, 19 dets - `068_..._frame_0018`

### Frames sem nenhuma deteccao (4)
- `019_..._frame_0012`
- `019_..._frame_0041`
- `026_..._frame_0033`
- `036_..._frame_0029` (video com cena em 1a pessoa)

### Frames com adulto_ok (36 - raros)
Salvos integralmente no `analisar_resultados.py` (saida completa). Exemplos:
- `007_..._frame_0007` (3 classes: afogamento + crianca_ok + adulto_ok, conf=0.659)
- `019_..._frame_0042` (afogamento + adulto_ok, conf=0.860 - alta confianca)

## Recomendacao para revisao no Roboflow

Ordem de prioridade da revisao manual:

1. **Re-anotar a classe `afogamento`** numa amostra de 20-30 frames para confirmar se o modelo aprendeu o "padrao de cena inteira" - se confirmado, considerar refazer as anotacoes desta classe no dataset com bboxes mais especificas (apenas a pessoa/area de afogamento real).
2. **Validar os 36 frames com `adulto_ok`** - se forem corretos, reforcar essa classe no proximo treino. Se forem confusoes com `crianca_ok`, ajustar dataset.
3. **Os 4 frames sem deteccao** - entender por que (cena dificil? frame inicio/fim do video?).
4. **Top 5 com baixa confianca** - candidatos a falso positivo, ja extraidos acima.
5. **Top 5 com muitas deteccoes** - verificar se sao bboxes proliferadas de piscinas.

## Saidas geradas
```
AutoAnotation/output/
  videos/               73 videos (147 arquivos com streams nao-mesclados)
  frames/<nome_video>/  frames originais por video
  labels/<nome_video>/  labels .txt por video
  roboflow_import/
    images/             1800 .jpg consolidados
    labels/             1800 .txt consolidados
    classes.txt
    relatorio.csv       relatorio por frame
    prioritarios_revisar.txt  1783 frames com classe afogamento
```

## Notas tecnicas
- ffmpeg do conda-forge esta com DLL quebrada nesta instalacao (exit code 0xC0000139). Foi substituido por `cv2.VideoCapture` para extracao de frames. Isso evita a dependencia externa.
- yt-dlp tambem usa ffmpeg para mesclar video+audio - como o ffmpeg falhou, os 73 videos ficaram divididos em 147 arquivos (streams separados). O pipeline detecta isso via `selecionar_videos()` e processa apenas o stream com video, ignorando o audio-only.
- Frame extraction usa `CAP_PROP_POS_FRAMES` (seek por indice) para precisao do intervalo. fps medio dos videos: 59.9.
- Tempo de execucao pos-download: ~6 minutos (yolo26n em GPU RTX 4050).
