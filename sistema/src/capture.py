"""Camada de aquisição — captura do fluxo RTSP em thread dedicada.

Mantém sempre o frame mais recente disponível, descartando frames antigos
quando a inferência é mais lenta que a taxa da câmera. Isso evita o acúmulo
de latência no buffer interno do OpenCV, o que comprometeria a hipótese H2
(latência < 10 s entre captura e alerta).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    capture_ts: float   # timestamp de captura (base para a medição de H2)
    index: int


class VideoSource:
    """Fonte de vídeo unificada: RTSP (tempo real) ou arquivo (validação).

    No modo RTSP, uma thread lê continuamente e guarda apenas o último frame.
    No modo arquivo, a leitura é sequencial e síncrona, para que todos os
    frames dos cenários gravados sejam processados (protocolo da seção 8.6).
    """

    def __init__(self, uri: str, realtime: bool = False):
        self.uri = uri
        self.is_live = uri.lower().startswith(("rtsp://", "http://", "https://"))
        self.realtime = realtime and not self.is_live
        self._cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {uri}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0

        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._running = False
        self._index = 0
        self._start_mono: float | None = None
        self._start_wall: float | None = None

        if self.is_live:
            self._running = True
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ RTSP
    def _reader(self) -> None:
        while self._running:
            ok, img = self._cap.read()
            if not ok:
                time.sleep(0.05)  # reconexão simples; robustecer se necessário
                continue
            frame = Frame(image=img, capture_ts=time.time(), index=self._index)
            self._index += 1
            with self._lock:
                self._latest = frame

    # ------------------------------------------------------------------- API
    def read(self) -> Frame | None:
        """Retorna o próximo frame a processar (ou None ao fim do vídeo)."""
        if self.is_live:
            with self._lock:
                frame = self._latest
                self._latest = None  # consome; evita reprocessar o mesmo frame
            return frame

        if self.realtime:
            return self._read_realtime()

        # Modo offline: processa todos os frames o mais rápido possível. O
        # capture_ts usa o tempo de vídeo (início + idx/fps), não o relógio de
        # parede, para que a janela temporal da decisão cubra 3 s de vídeo — a
        # velocidade de processamento não deve alterar a lógica de detecção.
        ok, img = self._cap.read()
        if not ok:
            return None
        if self._start_wall is None:
            self._start_wall = time.time()
        idx = self._index
        self._index += 1
        return Frame(image=img, capture_ts=self._start_wall + idx / self.fps, index=idx)

    def _read_realtime(self) -> Frame | None:
        """Modo validação com relógio real (H2): entrega frames na cadência
        nominal do arquivo, emulando uma câmera — se a inferência atrasa,
        frames intermediários são descartados (como no modo RTSP); se adianta,
        espera o instante de captura do frame. O capture_ts corresponde ao
        momento em que uma câmera teria capturado aquele frame."""
        if self._start_mono is None:
            self._start_mono = time.monotonic()
            self._start_wall = time.time()
        alvo = int((time.monotonic() - self._start_mono) * self.fps)
        while True:
            ok, img = self._cap.read()
            if not ok:
                return None
            idx = self._index
            self._index += 1
            if idx >= alvo:
                break
        espera = (self._start_mono + idx / self.fps) - time.monotonic()
        if espera > 0:
            time.sleep(espera)
        return Frame(image=img, capture_ts=self._start_wall + idx / self.fps, index=idx)

    def release(self) -> None:
        self._running = False
        if self.is_live:
            self._thread.join(timeout=1.0)
        self._cap.release()
