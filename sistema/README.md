# Sistema de Detecção de Afogamentos em Piscinas

Implementação da arquitetura em três camadas descrita na seção 8.5 do TCC:
aquisição (RTSP) → processamento (YOLO + janela deslizante de 3 s) → notificação sonora.

## Estrutura

```
sistema/
├── config.yaml            # parâmetros do sistema (janela, limiares, classes)
├── requirements.txt
├── models/                # colocar aqui o best.pt do treinamento
├── assets/                # alarm.wav (som do alerta)
├── videos/                # cenários gravados para validação (seção 8.6)
├── logs/                  # CSVs de métricas (H2 e H3)
└── src/
    ├── capture.py         # camada de aquisição (RTSP em thread / vídeo)
    ├── detector.py        # inferência YOLO + ROI da classe piscinas
    ├── decision.py        # janela 3 s, limiar 60%, cooldown 30 s
    ├── notifier.py        # alerta sonoro não bloqueante (winsound)
    ├── metrics.py         # registro de latência (H2) e janelas/alertas (H3)
    ├── analyze_latency.py # média, desvio e P95 das latências
    └── main.py            # orquestração
```

## Instalação

Usa o mesmo ambiente conda do projeto (já contém todas as dependências):

```bash
conda activate tcc
```

Copie os pesos treinados (ex.: `runs/yolo11n/weights/best.pt`) para `models/`
e ajuste `config.yaml`. As classes são referenciadas por **nome** no config;
os ids são resolvidos em runtime a partir de `model.names` do checkpoint.

## Execução

```bash
# Tempo real (câmera IP)
python -m src.main --config config.yaml --show

# Validação com cenário gravado (seção 8.6)
python -m src.main --source videos/cenario_06.mp4 --scenario cenario_06_water_milling

# Análise de latência (H2)
python -m src.analyze_latency --logs logs/
```

## Mapeamento com as hipóteses

| Hipótese | Onde é medida |
| --- | --- |
| H1 (acurácia/recall) | avaliação do modelo no conjunto de teste (Ultralytics `val`) + cenários positivos |
| H2 (latência < 10 s) | `metrics.py` registra ts_captura → ts_emissao; `analyze_latency.py` reporta média/DP/P95 |
| H3 (FP < 15%) | `resumo_*.csv`: alertas emitidos ÷ janelas avaliadas nos cenários negativos |
