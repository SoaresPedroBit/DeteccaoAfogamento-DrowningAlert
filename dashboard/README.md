# Dashboard — Detecção de Afogamentos

Interface web de apoio ao operador (salva-vidas), implementada conforme o
[documento de decisões técnicas](decisoes-dashboard-tcc.md): FastAPI + SPA única
em HTML/Chart.js (D-08, opção B), SQLite (D-03), stream MJPEG (D-02) e SSE (D-07).

## Arquitetura

O dashboard **não executa inferência** (D-01). O processo de inferência
(`sistema/`) publica três artefatos que a API consome:

| Artefato | Produzido por | Conteúdo |
|---|---|---|
| `data/latest.jpg` | `sistema/src/dashboard_bridge.py` | último frame anotado (escrita atômica) |
| `data/status.json` | idem | FPS, modelo, tempo de inferência, último frame |
| `data/eventos.db` | idem (+ API no PATCH de status) | eventos (D-04) e amostras de desempenho |
| `snapshots/{ano}/{mes}/{id}.jpg` | idem | snapshot anotado do alerta (D-05), com blur facial (D-10) |

## Como executar

```bash
conda activate tcc
pip install -r requirements.txt          # fastapi + uvicorn (uma vez)

# Terminal 1 — inferência (grava eventos e publica o stream):
cd ../sistema
python -m src.main --config config.yaml  # bloco dashboard: do config.yaml

# Terminal 2 — API + interface:
cd ../dashboard
uvicorn app:app --host 0.0.0.0 --port 8000
```

Acesse `http://localhost:8000`. Autenticação HTTP Basic (D-11):
usuário/senha `admin`/`admin` por padrão — troque via variáveis de ambiente
`DASHBOARD_USER` e `DASHBOARD_PASS`.

### Demonstração sem câmera

```bash
python gerar_dados_demo.py     # popula eventos, métricas e snapshots sintéticos
uvicorn app:app --port 8000
```

O script recusa rodar sobre um banco com dados reais (use `--forcar` só em teste).
Para zerar: apague `data/` e `snapshots/`.

## Telas

- **Ao vivo** — player MJPEG, faixa de status (câmera, FPS, modelo, uptime),
  últimos 5 eventos via SSE, banner + som quando `classe = afogamento`.
- **Análise** — gráficos G1–G6 mapeados às hipóteses (seção 8), filtro de
  período e de modelo; todo gráfico exporta PNG e CSV.
- **Histórico** — tabela paginada com filtros; clique abre o painel com
  snapshot, metadados e a revisão do operador (D-06: ocorrência real /
  falso positivo / observação), que alimenta a precisão em operação.

## Retenção (D-09)

Snapshots com mais de `DASHBOARD_RETENCAO_DIAS` dias (padrão **30** — prazo
definitivo pendente do protocolo CEP) são removidos por rotina diária da API;
os metadados numéricos do evento são preservados e `caminho_snapshot` é
marcado como `expirado`.

## Observações de implementação

- SQLite em modo WAL: inferência escreve, API lê; a única escrita da API é o
  `PATCH /api/eventos/{id}/status` (seção 13 do documento).
- O esquema SQL existe em `db.py` e em `sistema/src/dashboard_bridge.py`
  (processos independentes) — **manter os dois em sincronia**.
- Chart.js servido localmente (`static/vendor/`), sem dependência de CDN:
  o dashboard opera em rede local.
