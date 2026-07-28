"""Acesso ao banco de eventos do dashboard (D-03/D-04).

SQLite em modo WAL: o processo de inferência escreve (eventos, métricas de
desempenho) e a API lê — a única escrita da API é o PATCH de status (D-06).
O esquema é idêntico ao de sistema/src/dashboard_bridge.py; manter em sincronia.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "eventos.db"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"

SCHEMA = """
CREATE TABLE IF NOT EXISTS eventos (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_inicio    TEXT    NOT NULL,
    timestamp_alerta    TEXT    NOT NULL,
    camera_id           TEXT    NOT NULL,
    classe              TEXT    NOT NULL,
    confianca_media     REAL    NOT NULL,
    confianca_maxima    REAL    NOT NULL,
    frames_positivos    INTEGER NOT NULL,
    frames_janela       INTEGER NOT NULL,
    latencia_ms         INTEGER NOT NULL,
    modelo              TEXT    NOT NULL,
    caminho_snapshot    TEXT    NOT NULL,
    status              TEXT    DEFAULT 'pendente',
    revisado_em         TEXT,
    observacao          TEXT
);
CREATE INDEX IF NOT EXISTS idx_eventos_timestamp ON eventos(timestamp_alerta);
CREATE INDEX IF NOT EXISTS idx_eventos_status    ON eventos(status);

CREATE TABLE IF NOT EXISTS metricas_desempenho (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    modelo              TEXT    NOT NULL,
    fps                 REAL    NOT NULL,
    inferencia_ms_media REAL    NOT NULL,
    inferencia_ms_p95   REAL,
    frames              INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_desempenho_timestamp ON metricas_desempenho(timestamp);
"""


@contextmanager
def connect(readonly: bool = False):
    """Conexão com commit e close garantidos (o `with` nativo do sqlite3 não fecha)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    if readonly:
        conn.execute("PRAGMA query_only=ON")
    try:
        yield conn
        if not readonly:
            conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
