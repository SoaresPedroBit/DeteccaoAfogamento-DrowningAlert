"""API do dashboard (FastAPI) — seções 2, 6, 10 e 11 do documento de decisões.

Não executa inferência (D-01): consome o que o processo de inferência publica —
`data/latest.jpg` (frame anotado), `data/status.json` e `data/eventos.db`.

Uso:
  conda activate tcc
  uvicorn app:app --host 0.0.0.0 --port 8000

Autenticação (D-11): HTTP Basic. Credenciais via variáveis de ambiente
DASHBOARD_USER e DASHBOARD_PASS (padrão admin/admin — trocar em operação).
Retenção de snapshots (D-09): DASHBOARD_RETENCAO_DIAS (padrão 30; prazo
definitivo pendente de confirmação com o protocolo do CEP).
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import os
import secrets
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
LATEST_FRAME = db.DATA_DIR / "latest.jpg"
STATUS_JSON = db.DATA_DIR / "status.json"

# Taxa de entrega ao navegador (D-02). Deve acompanhar dashboard.stream_fps do
# config.yaml do sistema — o menor dos dois é o teto efetivo no navegador.
STREAM_FPS = float(os.environ.get("DASHBOARD_STREAM_FPS", "15"))
CAMERA_TIMEOUT_S = 5                 # seção 13: indicador de desconexão
RETENCAO_DIAS = int(os.environ.get("DASHBOARD_RETENCAO_DIAS", "30"))

AUTH_USER = os.environ.get("DASHBOARD_USER", "admin")
AUTH_PASS = os.environ.get("DASHBOARD_PASS", "admin")
# Desligar APENAS em desenvolvimento local (D-11 exige autenticação em operação)
AUTH_ATIVA = os.environ.get("DASHBOARD_AUTH", "on").lower() not in ("off", "0", "false")

app = FastAPI(title="Dashboard — Detecção de Afogamentos", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------- auth
@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not AUTH_ATIVA:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    ok = False
    if header.startswith("Basic "):
        try:
            user, _, pwd = base64.b64decode(header[6:]).decode().partition(":")
            ok = secrets.compare_digest(user, AUTH_USER) and secrets.compare_digest(pwd, AUTH_PASS)
        except Exception:
            ok = False
    if not ok:
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="dashboard"'})
    return await call_next(request)


# ------------------------------------------------------------------ startup
@app.on_event("startup")
async def startup() -> None:
    db.init()
    asyncio.create_task(_rotina_retencao())


async def _rotina_retencao() -> None:
    """D-09: remove snapshots expirados, preservando os metadados do evento."""
    while True:
        limite = (datetime.now() - timedelta(days=RETENCAO_DIAS)).isoformat()
        try:
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT id, caminho_snapshot FROM eventos "
                    "WHERE timestamp_alerta < ? AND caminho_snapshot NOT IN ('', 'expirado')",
                    (limite,),
                ).fetchall()
                for row in rows:
                    caminho = db.BASE_DIR / row["caminho_snapshot"]
                    if caminho.is_file():
                        caminho.unlink()
                    conn.execute(
                        "UPDATE eventos SET caminho_snapshot='expirado' WHERE id=?",
                        (row["id"],),
                    )
            if rows:
                print(f"[retenção] {len(rows)} snapshot(s) expirados removidos")
        except Exception as exc:
            print(f"[retenção] erro: {exc}")
        await asyncio.sleep(24 * 3600)


# ------------------------------------------------------------------- stream
async def _mjpeg_generator():
    intervalo = 1.0 / STREAM_FPS
    ultima_mtime = 0.0
    while True:
        if LATEST_FRAME.is_file():
            mtime = LATEST_FRAME.stat().st_mtime
            if mtime != ultima_mtime:
                ultima_mtime = mtime
                try:
                    dados = LATEST_FRAME.read_bytes()
                except OSError:
                    dados = b""  # arquivo sendo substituído; tenta no próximo ciclo
                if dados:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           + f"Content-Length: {len(dados)}\r\n\r\n".encode()
                           + dados + b"\r\n")
        await asyncio.sleep(intervalo)


@app.get("/api/stream")
async def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


# ------------------------------------------------------------------- status
@app.get("/api/status")
async def status():
    info: dict = {}
    if STATUS_JSON.is_file():
        try:
            info = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            info = {}
    ultimo = float(info.get("ultimo_frame_ts", 0.0))
    agora = time.time()
    conectada = (agora - ultimo) < CAMERA_TIMEOUT_S
    iniciado = float(info.get("iniciado_em", 0.0))
    return {
        "camera_conectada": conectada,
        "camera_id": info.get("camera_id", "—"),
        "fonte": info.get("fonte", "—"),
        "fps": info.get("fps", 0.0) if conectada else 0.0,
        "modelo": info.get("modelo", "—"),
        "inferencia_ms_media": info.get("inferencia_ms_media", 0.0),
        "uptime_s": int(agora - iniciado) if conectada and iniciado else 0,
        "ultimo_frame_ha_s": round(agora - ultimo, 1) if ultimo else None,
        "retencao_dias": RETENCAO_DIAS,
    }


# ------------------------------------------------------------------ eventos
def _filtros_eventos(de, ate, classe, status_, modelo):
    where, params = [], []
    if de:
        where.append("timestamp_alerta >= ?")
        params.append(de)
    if ate:
        where.append("timestamp_alerta <= ?")
        params.append(ate + ("T23:59:59" if len(ate) == 10 else ""))
    if classe:
        where.append("classe = ?")
        params.append(classe)
    if status_:
        where.append("status = ?")
        params.append(status_)
    if modelo:
        where.append("modelo = ?")
        params.append(modelo)
    return ("WHERE " + " AND ".join(where)) if where else "", params


@app.get("/api/eventos")
async def listar_eventos(
    de: str | None = None,
    ate: str | None = None,
    classe: str | None = None,
    status: str | None = None,
    modelo: str | None = None,
    pagina: int = Query(1, ge=1),
    por_pagina: int = Query(20, ge=1, le=200),
):
    clausula, params = _filtros_eventos(de, ate, classe, status, modelo)
    with db.connect(readonly=True) as conn:
        total = conn.execute(f"SELECT COUNT(*) c FROM eventos {clausula}", params).fetchone()["c"]
        rows = conn.execute(
            f"SELECT * FROM eventos {clausula} ORDER BY timestamp_alerta DESC LIMIT ? OFFSET ?",
            params + [por_pagina, (pagina - 1) * por_pagina],
        ).fetchall()
    return {
        "total": total,
        "pagina": pagina,
        "por_pagina": por_pagina,
        "eventos": [dict(r) for r in rows],
    }


@app.get("/api/eventos/stream")
async def eventos_stream(request: Request):
    """D-07: SSE de novos alertas — fluxo unidirecional servidor → cliente."""
    async def gerador():
        with db.connect(readonly=True) as conn:
            row = conn.execute("SELECT MAX(id) m FROM eventos").fetchone()
        ultimo_id = row["m"] or 0
        yield "retry: 3000\n\n"
        while True:
            if await request.is_disconnected():
                return
            with db.connect(readonly=True) as conn:
                novos = conn.execute(
                    "SELECT * FROM eventos WHERE id > ? ORDER BY id", (ultimo_id,)
                ).fetchall()
            for row in novos:
                ultimo_id = row["id"]
                yield f"event: evento\ndata: {json.dumps(dict(row), ensure_ascii=False)}\n\n"
            if not novos:
                yield ": heartbeat\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        gerador(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/eventos/{evento_id}")
async def detalhe_evento(evento_id: int):
    with db.connect(readonly=True) as conn:
        row = conn.execute("SELECT * FROM eventos WHERE id=?", (evento_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Evento não encontrado")
    return dict(row)


class StatusPatch(BaseModel):
    status: str          # confirmado | falso_positivo | pendente
    observacao: str | None = None


@app.patch("/api/eventos/{evento_id}/status")
async def confirmar_evento(evento_id: int, patch: StatusPatch):
    """D-06: classificação manual do operador — ground truth em operação."""
    if patch.status not in ("confirmado", "falso_positivo", "pendente"):
        raise HTTPException(422, "status deve ser confirmado, falso_positivo ou pendente")
    revisado = None if patch.status == "pendente" else datetime.now().isoformat(timespec="seconds")
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE eventos SET status=?, revisado_em=?, observacao=? WHERE id=?",
            (patch.status, revisado, patch.observacao, evento_id),
        )
    if cur.rowcount == 0:
        raise HTTPException(404, "Evento não encontrado")
    return {"ok": True, "id": evento_id, "status": patch.status, "revisado_em": revisado}


# ----------------------------------------------------------------- métricas
def _bucket_sql(de: str | None, ate: str | None) -> str:
    """Agregação adaptativa: minuto (≤1 dia), hora (≤7 dias) ou dia."""
    try:
        inicio = datetime.fromisoformat(de) if de else None
        fim = datetime.fromisoformat(ate) if ate else datetime.now()
        dias = (fim - inicio).days if inicio else 9999
    except ValueError:
        dias = 9999
    if dias <= 1:
        return "substr(timestamp, 1, 16)"   # YYYY-MM-DDTHH:MM
    if dias <= 7:
        return "substr(timestamp, 1, 13)"   # YYYY-MM-DDTHH
    return "substr(timestamp, 1, 10)"       # YYYY-MM-DD


@app.get("/api/metricas")
async def metricas(de: str | None = None, ate: str | None = None, modelo: str | None = None):
    """Séries agregadas para os gráficos G1–G6 (seção 8)."""
    clausula_ev, params_ev = _filtros_eventos(de, ate, None, None, modelo)
    cl_ev_afog = (clausula_ev + (" AND " if clausula_ev else "WHERE ") + "classe='afogamento'")

    where_perf, params_perf = [], []
    if de:
        where_perf.append("timestamp >= ?")
        params_perf.append(de)
    if ate:
        where_perf.append("timestamp <= ?")
        params_perf.append(ate + ("T23:59:59" if len(ate) == 10 else ""))
    if modelo:
        where_perf.append("modelo = ?")
        params_perf.append(modelo)
    cl_perf = ("WHERE " + " AND ".join(where_perf)) if where_perf else ""
    bucket = _bucket_sql(de, ate)

    with db.connect(readonly=True) as conn:
        # G1 — latência de detecção (ms) por evento de afogamento
        g1 = [r["latencia_ms"] for r in conn.execute(
            f"SELECT latencia_ms FROM eventos {cl_ev_afog}", params_ev)]

        # G2 — distribuição de confiança por resultado da revisão (D-06)
        g2 = {"confirmado": [], "falso_positivo": [], "pendente": []}
        for r in conn.execute(
                f"SELECT status, confianca_media FROM eventos {cl_ev_afog}", params_ev):
            g2.setdefault(r["status"] or "pendente", []).append(r["confianca_media"])

        # G3 — FPS ao longo do tempo, por modelo
        g3 = [dict(r) for r in conn.execute(
            f"SELECT {bucket} periodo, modelo, AVG(fps) fps "
            f"FROM metricas_desempenho {cl_perf} "
            "GROUP BY periodo, modelo ORDER BY periodo", params_perf)]

        # G4 — tempo de inferência por frame, comparativo entre modelos
        g4 = []
        modelos = [r["modelo"] for r in conn.execute(
            f"SELECT DISTINCT modelo FROM metricas_desempenho {cl_perf}", params_perf)]
        for m in modelos:
            valores = [r["inferencia_ms_media"] for r in conn.execute(
                f"SELECT inferencia_ms_media FROM metricas_desempenho {cl_perf}"
                + (" AND " if cl_perf else "WHERE ") + "modelo=?",
                params_perf + [m])]
            if valores:
                g4.append({
                    "modelo": m,
                    "media": round(statistics.fmean(valores), 2),
                    "desvio": round(statistics.pstdev(valores), 2) if len(valores) > 1 else 0.0,
                    "minimo": round(min(valores), 2),
                    "maximo": round(max(valores), 2),
                    "amostras": len(valores),
                })

        # G5 — eventos de afogamento por hora do dia
        g5 = {int(r["hora"]): r["c"] for r in conn.execute(
            f"SELECT substr(timestamp_alerta, 12, 2) hora, COUNT(*) c "
            f"FROM eventos {cl_ev_afog} GROUP BY hora", params_ev)}

        # G6 — precisão e taxa de falsos positivos por dia (revisões D-06)
        g6 = []
        por_dia = conn.execute(
            f"SELECT substr(timestamp_alerta, 1, 10) dia, "
            "SUM(status='confirmado') vp, SUM(status='falso_positivo') fp "
            f"FROM eventos {cl_ev_afog} GROUP BY dia ORDER BY dia", params_ev).fetchall()
        horas_por_dia = {r["dia"]: r["h"] for r in conn.execute(
            f"SELECT substr(timestamp, 1, 10) dia, "
            "SUM(frames) * 1.0 / NULLIF(AVG(fps), 0) / 3600.0 h "
            f"FROM metricas_desempenho {cl_perf} GROUP BY dia", params_perf)}
        for r in por_dia:
            vp, fp = r["vp"] or 0, r["fp"] or 0
            horas = horas_por_dia.get(r["dia"])
            g6.append({
                "dia": r["dia"],
                "precisao": round(vp / (vp + fp), 4) if (vp + fp) else None,
                "fp_por_hora": round(fp / horas, 3) if horas else None,
                "confirmados": vp,
                "falsos_positivos": fp,
            })

        # Resumo para os cartões da tela de análise
        tot = conn.execute(
            f"SELECT COUNT(*) c, SUM(status='confirmado') vp, "
            f"SUM(status='falso_positivo') fp, SUM(status='pendente') pend "
            f"FROM eventos {cl_ev_afog}", params_ev).fetchone()

    vp, fp = tot["vp"] or 0, tot["fp"] or 0
    return {
        "resumo": {
            "total_eventos": tot["c"],
            "confirmados": vp,
            "falsos_positivos": fp,
            "pendentes": tot["pend"] or 0,
            "precisao": round(vp / (vp + fp), 4) if (vp + fp) else None,
            "latencia_media_ms": round(statistics.fmean(g1), 1) if g1 else None,
        },
        "g1_latencias_ms": g1,
        "g2_confianca_por_status": g2,
        "g3_fps_por_periodo": g3,
        "g4_inferencia_por_modelo": g4,
        "g5_eventos_por_hora": g5,
        "g6_precisao_por_dia": g6,
    }


# ---------------------------------------------------------------- exportação
@app.get("/api/exportar")
async def exportar(
    de: str | None = None,
    ate: str | None = None,
    classe: str | None = None,
    status: str | None = None,
    modelo: str | None = None,
):
    clausula, params = _filtros_eventos(de, ate, classe, status, modelo)
    with db.connect(readonly=True) as conn:
        rows = conn.execute(
            f"SELECT * FROM eventos {clausula} ORDER BY timestamp_alerta DESC", params
        ).fetchall()
    buffer = io.StringIO()
    campos = rows[0].keys() if rows else [
        "id", "timestamp_inicio", "timestamp_alerta", "camera_id", "classe",
        "confianca_media", "confianca_maxima", "frames_positivos", "frames_janela",
        "latencia_ms", "modelo", "caminho_snapshot", "status", "revisado_em", "observacao",
    ]
    writer = csv.DictWriter(buffer, fieldnames=campos)
    writer.writeheader()
    writer.writerows([dict(r) for r in rows])
    nome = f"eventos_{datetime.now():%Y%m%d_%H%M%S}.csv"
    return Response(
        buffer.getvalue().encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{nome}"'},
    )


# ------------------------------------------------------------------ estático
db.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=db.SNAPSHOTS_DIR), name="snapshots")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="spa")
