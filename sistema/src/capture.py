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

    def __init__(self, uri: str):
        self.uri = uri
        self.is_live = uri.lower().startswith(("rtsp://", "http://", "https://"))
        self._cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {uri}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0

        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._running = False
        self._index = 0

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

        ok, img = self._cap.read()
        if not ok:
            return None
        frame = Frame(image=img, capture_ts=time.time(), index=self._index)
        self._index += 1
        return frame

    def release(self) -> None:
        self._running = False
        if self.is_live:
            self._thread.join(timeout=1.0)
        self._cap.release()
