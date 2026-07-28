# Comandos úteis — TCC Detecção de Afogamentos

Referência rápida de todos os comandos do projeto (PowerShell, Windows).

> **Sobre o Python:** os comandos usam o caminho completo do ambiente `tcc`
> (`C:\Users\soare\miniconda3\envs\tcc\python.exe`), que funciona em qualquer
> terminal. Se preferir usar `conda activate tcc` + `python`, habilite o conda
> no PowerShell uma única vez e reabra o terminal:
>
> ```powershell
> C:\Users\soare\miniconda3\Scripts\conda.exe init powershell
> ```

```powershell
# atalho usado nos exemplos abaixo (vale só para a sessão atual do terminal)
$py = "C:\Users\soare\miniconda3\envs\tcc\python.exe"
```

---

## 1. Dashboard (interface web)

```powershell
cd C:\Users\soare\Documents\Faculdade\Faculdade\TCC\dashboard

# iniciar a API + interface (http://localhost:8000 — login admin/admin)
& $py -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Variáveis de ambiente (definir ANTES do uvicorn, no mesmo terminal):

```powershell
$env:DASHBOARD_USER = "operador"      # usuário do login (padrão: admin)
$env:DASHBOARD_PASS = "senha-forte"   # senha do login  (padrão: admin)
$env:DASHBOARD_AUTH = "off"           # desliga o login — SÓ em desenvolvimento
$env:DASHBOARD_RETENCAO_DIAS = "30"   # prazo de retenção dos snapshots (D-09)
$env:DASHBOARD_STREAM_FPS = "15"      # taxa de entrega do stream ao navegador
```

Parar: `Ctrl+C` no terminal do uvicorn.

---

## 2. Inferência (é ela que alimenta o dashboard)

Sempre em outro terminal, com o dashboard rodando em paralelo.

```powershell
cd C:\Users\soare\Documents\Faculdade\Faculdade\TCC\sistema
$py = "C:\Users\soare\miniconda3\envs\tcc\python.exe"
```

### Câmera IP (operação real)

```powershell
# ajuste source.uri no config.yaml (rtsp://usuario:senha@ip:554/stream1) e:
& $py -m src.main --config config.yaml
```

### Um vídeo — modo VALIDAÇÃO (latências válidas para a H2)

`--realtime` reproduz o vídeo na taxa nominal, emulando a câmera.
Use este modo para toda medição que vai para a monografia.

```powershell
& $py -m src.main --config config.yaml --source "videos/video2.mp4" --scenario video2 --realtime
```

### Um vídeo — modo RÁPIDO (só para ver SE detecta)

Sem `--realtime` o vídeo é processado na velocidade máxima da GPU.
As latências saem distorcidas — não usar para a H2.

```powershell
& $py -m src.main --config config.yaml --source "videos/video2.mp4" --scenario video2
```

### Lote de vídeos (um após o outro)

```powershell
# todos os .mp4 da pasta videos/:
foreach ($v in Get-ChildItem videos\*.mp4) {
    & $py -m src.main --config config.yaml --source "videos/$($v.Name)" --scenario $v.BaseName --realtime
}

# ou só alguns, escolhidos à mão:
foreach ($v in "video2.mp4", "video3.mp4") {
    & $py -m src.main --config config.yaml --source "videos/$v" --scenario ([IO.Path]::GetFileNameWithoutExtension($v)) --realtime
}
```

### Opções extras

```powershell
--show        # abre janela com as detecções desenhadas (debug visual)
--scenario X  # nome do cenário nos CSVs de sistema/logs/ (seção 8.6)
```

Para comparar modelos no dashboard (G3/G4): rode a bateria com um peso,
troque `model.weights` e `model.nome` no `config.yaml` (ex.: yolo11n → yolo26n)
e rode a bateria de novo. Os dois convivem no mesmo banco.

---

## 3. Dados do dashboard

```powershell
cd C:\Users\soare\Documents\Faculdade\Faculdade\TCC\dashboard

# ZERAR tudo (eventos, métricas e snapshots) — pare o uvicorn antes!
Remove-Item -Recurse -Force data, snapshots

# popular com dados sintéticos p/ demonstrar a interface sem câmera
& $py gerar_dados_demo.py            # recusa rodar sobre dados reais
& $py gerar_dados_demo.py --forcar   # sobrescreve mesmo assim (cuidado)
```

Exportações (também disponíveis pela interface):
- CSV dos eventos filtrados: botão **Exportar CSV** na tela Histórico.
- PNG/CSV de cada gráfico: botões no canto de cada card da tela Análise.

---

## 4. Treino e dataset (scripts/)

```powershell
cd C:\Users\soare\Documents\Faculdade\Faculdade\TCC
$py = "C:\Users\soare\miniconda3\envs\tcc\python.exe"

& $py scripts\01_verificar_ambiente.py        # valida GPU/CUDA/pacotes
$env:ROBOFLOW_API_KEY = "<sua-key>"
& $py scripts\02_baixar_dataset.py            # baixa dataset do Roboflow
& $py scripts\03_treinar.py                   # treina yolo11n e yolo26n
& $py scripts\03_treinar.py --modelo yolo11n  # treina só um
& $py scripts\04_comparar_modelos.py          # gera comparativo de métricas
& $py sistema\src\analyze_latency.py          # média/desvio/P95 das latências (H2)
```

---

## 5. Roteiro típico de uma sessão de validação

```powershell
# 1. (opcional) zerar o banco para uma bateria limpa      → seção 3
# 2. Terminal A: subir o dashboard                        → seção 1
# 3. Terminal B: rodar os cenários com --realtime         → seção 2
# 4. No navegador: revisar cada evento no Histórico
#    (Ocorrência real / Falso positivo) — alimenta G2/G6 e a precisão
# 5. Tela Análise: exportar os gráficos em PNG/CSV para a monografia
```
