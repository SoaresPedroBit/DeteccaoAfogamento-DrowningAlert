"""Ponte entre o processo de inferência e o dashboard (D-01).

O dashboard não executa inferência própria: este módulo publica, do lado da
inferência, os três artefatos que a API consome:

  1. ``data/latest.jpg``  — último frame anotado (buffer de um slot, escrita
     atômica), codificado em thread separada para não bloquear o loop
     principal (mitigação da seção 13);
  2. ``data/status.json`` — FPS, modelo, tempo de inferência, último frame;
  3. ``data/eventos.db``  — eventos (D-04) e amostras de desempenho, com o
     snapshot anotado em ``snapshots/{ano}/{mes}/{id}.jpg`` (D-05) e
     anonimização facial opcional (D-10).

O esquema SQL é idêntico ao de dashboard/db.py; manter em sincronia.
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

_SCHEMA = """
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


class DashboardBridge:
    def __init__(
        self,
        data_dir: str,
        snapshots_dir: str,
        camera_id: str,
        modelo: str,
        fonte: str,
        window_size: int,
        drowning_class: str = "afogamento",
        stream_fps: float = 12.0,
        stream_quality: int = 70,
        stream_max_width: int = 1280,
        snapshot_quality: int = 85,
        perf_sample_seconds: float = 5.0,
        anonimizar_snapshots: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.camera_id = camera_id
        self.modelo = modelo
        self.fonte = fonte
        self.drowning_class = drowning_class
        self.stream_quality = int(stream_quality)
        self.stream_max_width = int(stream_max_width)
        self.snapshot_quality = int(snapshot_quality)
        self.perf_sample_seconds = float(perf_sample_seconds)
        self.iniciado_em = time.time()

        self.db_path = self.data_dir / "eventos.db"
        self._init_db()

        # janela paralela de confianças da classe afogamento (p/ D-04)
        self._confs: deque[float] = deque(maxlen=max(1, window_size))
        self.frames_janela = max(1, window_size)

        # acumuladores da amostra de desempenho
        self._inf_ms: list[float] = []
        self._amostra_inicio = time.time()
        self._ultimo_frame_ts = 0.0

        # anonimização facial (D-10)
        self._cascade = None
        if anonimizar_snapshots:
            arquivo = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if arquivo.is_file():
                self._cascade = cv2.CascadeClassifier(str(arquivo))
            else:
                print("[dashboard] cascade de faces não encontrado — snapshots sem anonimização")

        # publicador de frames em thread própria (um slot; frames excedentes caem)
        self._frame_lock = threading.Lock()
        self._frame_pendente = None
        self._frame_evento = threading.Event()
        self._parar = threading.Event()
        self._intervalo_stream = 1.0 / max(1.0, float(stream_fps))
        self._ultimo_publish = 0.0
        self._worker = threading.Thread(target=self._loop_publicacao, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ db
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------- loop por frame
    def observe(self, detections, inferencia_ms: float) -> None:
        """Chamado a cada frame: alimenta a janela de confianças e o desempenho."""
        confs = [d.conf for d in detections if d.cls_name == self.drowning_class]
        self._confs.append(max(confs) if confs else 0.0)

        self._inf_ms.append(inferencia_ms)
        self._ultimo_frame_ts = time.time()
        decorrido = self._ultimo_frame_ts - self._amostra_inicio
        if decorrido >= self.perf_sample_seconds:
            self._gravar_amostra(decorrido)

    def _gravar_amostra(self, decorrido: float) -> None:
        frames = len(self._inf_ms)
        fps = frames / decorrido if decorrido > 0 else 0.0
        media = statistics.fmean(self._inf_ms) if self._inf_ms else 0.0
        p95 = (statistics.quantiles(self._inf_ms, n=20)[-1]
               if frames >= 20 else max(self._inf_ms, default=0.0))
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO metricas_desempenho "
                "(timestamp, modelo, fps, inferencia_ms_media, inferencia_ms_p95, frames) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), self.modelo,
                 round(fps, 2), round(media, 2), round(p95, 2), frames),
            )
            conn.commit()
        finally:
            conn.close()

        self._escrever_status(fps, media)
        self._inf_ms.clear()
        self._amostra_inicio = time.time()

    def _escrever_status(self, fps: float, inf_media: float) -> None:
        status = {
            "camera_id": self.camera_id,
            "modelo": self.modelo,
            "fonte": self.fonte,
            "fps": round(fps, 2),
            "inferencia_ms_media": round(inf_media, 2),
            "iniciado_em": self.iniciado_em,
            "ultimo_frame_ts": self._ultimo_frame_ts,
        }
        tmp = self.data_dir / "status.json.tmp"
        tmp.write_text(json.dumps(status), encoding="utf-8")
        self._substituir(tmp, self.data_dir / "status.json")

    @staticmethod
    def _substituir(tmp: Path, destino: Path) -> None:
        """os.replace com retry: no Windows, falha com acesso negado se a API
        estiver lendo o destino naquele instante. Se persistir, o frame cai —
        o próximo sobrescreve."""
        for _ in range(10):
            try:
                os.replace(tmp, destino)
                return
            except PermissionError:
                time.sleep(0.02)

    # ------------------------------------------------------------ streaming
    def publish_frame(self, image) -> None:
        """Entrega o frame anotado ao worker; se ele estiver ocupado, o frame cai.

        O corte de taxa acontece AQUI, antes do image.copy(): assim o custo no
        loop de inferência é pago só stream_fps vezes por segundo, e não a cada
        frame inferido (seção 13 — o stream não pode comprometer a medição de H2).
        """
        agora = time.monotonic()
        if agora - self._ultimo_publish < self._intervalo_stream:
            return
        self._ultimo_publish = agora
        with self._frame_lock:
            self._frame_pendente = image.copy()
        self._frame_evento.set()

    def _loop_publicacao(self) -> None:
        destino = self.data_dir / "latest.jpg"
        tmp = self.data_dir / "latest.jpg.tmp"
        while not self._parar.is_set():
            try:
                if not self._frame_evento.wait(timeout=0.5):
                    continue
                self._frame_evento.clear()
                with self._frame_lock:
                    frame = self._frame_pendente
                    self._frame_pendente = None
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                if w > self.stream_max_width:
                    escala = self.stream_max_width / w
                    frame = cv2.resize(frame, (self.stream_max_width, int(h * escala)),
                                       interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, self.stream_quality])
                if ok:
                    tmp.write_bytes(buf.tobytes())
                    self._substituir(tmp, destino)
                # a taxa (D-02) é imposta na entrada, em publish_frame()
            except Exception as exc:
                # a publicação nunca derruba a thread: descarta o frame e segue
                print(f"[dashboard] publicação do frame falhou: {exc}")
                time.sleep(0.5)

    # -------------------------------------------------------------- eventos
    def record_event(self, annotated_image, state, emitted_ts: float) -> int:
        """Grava o evento (D-04) e o snapshot anotado (D-05) no momento do alerta."""
        confs_positivas = [c for c in self._confs if c > 0]
        ts_inicio = datetime.fromtimestamp(state.trigger_capture_ts)
        ts_alerta = datetime.fromtimestamp(emitted_ts)
        latencia_ms = max(0, int((emitted_ts - state.trigger_capture_ts) * 1000))

        conn = self._conn()
        try:
            cur = conn.execute(
                "INSERT INTO eventos (timestamp_inicio, timestamp_alerta, camera_id, "
                "classe, confianca_media, confianca_maxima, frames_positivos, "
                "frames_janela, latencia_ms, modelo, caminho_snapshot) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts_inicio.isoformat(timespec="milliseconds"),
                    ts_alerta.isoformat(timespec="milliseconds"),
                    self.camera_id,
                    self.drowning_class,
                    round(statistics.fmean(confs_positivas), 4) if confs_positivas else 0.0,
                    round(max(confs_positivas), 4) if confs_positivas else 0.0,
                    int(round(state.ratio * self.frames_janela)),
                    self.frames_janela,
                    latencia_ms,
                    self.modelo,
                    "",
                ),
            )
            evento_id = cur.lastrowid
            caminho_rel = self._salvar_snapshot(annotated_image, ts_alerta, evento_id)
            conn.execute("UPDATE eventos SET caminho_snapshot=? WHERE id=?",
                         (caminho_rel, evento_id))
            conn.commit()
        finally:
            conn.close()
        # espelha o clear() que o DecisionEngine faz na janela após o disparo
        self._confs.clear()
        return evento_id

    def _salvar_snapshot(self, image, ts: datetime, evento_id: int) -> str:
        pasta = self.snapshots_dir / f"{ts:%Y}" / f"{ts:%m}"
        pasta.mkdir(parents=True, exist_ok=True)
        destino = pasta / f"{evento_id}.jpg"

        snapshot = image.copy()
        if self._cascade is not None:
            snapshot = self._anonimizar(snapshot)
        cv2.imwrite(str(destino), snapshot,
                    [cv2.IMWRITE_JPEG_QUALITY, self.snapshot_quality])
        # caminho relativo à raiz do dashboard, no formato servido pela API
        return f"snapshots/{ts:%Y}/{ts:%m}/{evento_id}.jpg"

    def _anonimizar(self, image):
        """D-10: borra as regiões de rosto antes de persistir o snapshot."""
        cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(cinza, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(24, 24))
        for (x, y, w, h) in faces:
            regiao = image[y:y + h, x:x + w]
            k = max(9, (w // 4) | 1)  # kernel ímpar proporcional ao rosto
            image[y:y + h, x:x + w] = cv2.GaussianBlur(regiao, (k, k), 0)
        return image

    # ----------------------------------------------------------------- fim
    def close(self) -> None:
        decorrido = time.time() - self._amostra_inicio
        if self._inf_ms and decorrido > 0:
            self._gravar_amostra(decorrido)
        self._parar.set()
        self._frame_evento.set()
        self._worker.join(timeout=2.0)


def from_config(cfg: dict, config_path: Path, fonte: str, window_size: int):
    """Cria a ponte a partir do bloco ``dashboard:`` do config.yaml (ou None)."""
    dcfg = cfg.get("dashboard", {})
    if not dcfg.get("enabled", False):
        return None
    base = config_path.resolve().parent
    modelo = cfg["model"].get("nome") or Path(cfg["model"]["weights"]).stem
    return DashboardBridge(
        data_dir=base / dcfg.get("data_dir", "../dashboard/data"),
        snapshots_dir=base / dcfg.get("snapshots_dir", "../dashboard/snapshots"),
        camera_id=dcfg.get("camera_id", "cam01"),
        modelo=modelo,
        fonte=fonte,
        window_size=window_size,
        drowning_class=cfg["model"]["drowning_class"],
        stream_fps=dcfg.get("stream_fps", 12),
        stream_quality=dcfg.get("stream_quality", 70),
        stream_max_width=dcfg.get("stream_max_width", 1280),
        snapshot_quality=dcfg.get("snapshot_quality", 85),
        perf_sample_seconds=dcfg.get("perf_sample_seconds", 5),
        anonimizar_snapshots=dcfg.get("anonimizar_snapshots", True),
    )
