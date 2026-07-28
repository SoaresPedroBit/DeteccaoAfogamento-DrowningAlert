"""Camada de processamento (parte 2) — lógica de decisão temporal.

Implementa a janela deslizante de 3 segundos: um alerta é disparado quando a
classe `afogamento` aparece com confiança mínima em pelo menos 60% dos frames
da janela. Após o disparo, um período de inibição (cooldown) de 30 s evita
alertas repetidos para o mesmo evento (seção 8.5 do TCC).

A janela é baseada em TEMPO (via `capture_ts`), não em contagem de frames: a
razão é calculada sobre os frames cujos timestamps de captura caem nos últimos
`window_seconds`. Isso a torna robusta a descarte de frames — que ocorre tanto
na câmera ao vivo (mantém-se só o frame mais recente) quanto no modo `--realtime`
(quando a inferência não acompanha a taxa nominal do vídeo). Uma janela por
contagem de frames só representaria 3 s reais se nenhum frame fosse descartado.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class WindowState:
    """Resultado da avaliação de um frame pela lógica de decisão."""

    triggered: bool                 # alerta disparado neste frame
    ratio: float                    # fração de frames positivos na janela atual
    window_full: bool               # janela completa (já tem ~3 s de histórico)
    in_cooldown: bool
    trigger_capture_ts: float = 0.0 # ts de captura do 1º frame positivo do evento


@dataclass
class DecisionEngine:
    fps: float = 25.0
    window_seconds: float = 3.0
    trigger_ratio: float = 0.6
    cooldown_seconds: float = 30.0
    drowning_class: str = "afogamento"  # comparação por nome (model.names)
    conf_threshold: float = 0.5

    _window: deque = field(init=False)          # (capture_ts, positive)
    _window_start_ts: float | None = field(default=None, init=False)
    _last_alert_ts: float = field(default=-1e9, init=False)
    _first_positive_ts: float = field(default=0.0, init=False)
    windows_evaluated: int = field(default=0, init=False)  # denominador da H3

    def __post_init__(self) -> None:
        # valor nominal, mantido só para exibição/registro (não limita a janela)
        self.window_size = max(1, int(round(self.fps * self.window_seconds)))
        self._window = deque()

    def update(self, detections, capture_ts: float, now: float) -> WindowState:
        positive = any(
            d.cls_name == self.drowning_class and d.conf >= self.conf_threshold
            for d in detections
        )

        if positive and not self._window_has_positive():
            self._first_positive_ts = capture_ts  # marca o início do evento (H2)

        if self._window_start_ts is None:
            self._window_start_ts = capture_ts
        self._window.append((capture_ts, positive))

        # descarta os frames cujo timestamp saiu dos últimos `window_seconds`
        limite = capture_ts - self.window_seconds
        while self._window and self._window[0][0] < limite:
            self._window.popleft()

        # janela cheia quando já há `window_seconds` de histórico acumulado
        window_full = (capture_ts - self._window_start_ts) >= self.window_seconds
        positivos = sum(p for _, p in self._window)
        ratio = positivos / len(self._window) if self._window else 0.0
        in_cooldown = (now - self._last_alert_ts) < self.cooldown_seconds

        if window_full:
            self.windows_evaluated += 1

        triggered = window_full and ratio >= self.trigger_ratio and not in_cooldown
        if triggered:
            self._last_alert_ts = now
            self._window.clear()          # reinicia o histórico após o disparo
            self._window_start_ts = None

        return WindowState(
            triggered=triggered,
            ratio=ratio,
            window_full=window_full,
            in_cooldown=in_cooldown,
            trigger_capture_ts=self._first_positive_ts,
        )

    def _window_has_positive(self) -> bool:
        return any(p for _, p in self._window)
