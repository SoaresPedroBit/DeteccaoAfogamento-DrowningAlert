"""Camada de processamento (parte 2) — lógica de decisão temporal.

Implementa a janela deslizante de 3 segundos (~75 frames a 25 FPS):
um alerta é disparado quando a classe `afogamento` aparece com confiança
mínima em pelo menos 60% dos frames da janela. Após o disparo, um período
de inibição (cooldown) de 30 s evita alertas repetidos para o mesmo evento
(seção 8.5 do TCC).
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

    _window: deque = field(init=False)
    _last_alert_ts: float = field(default=-1e9, init=False)
    _first_positive_ts: float = field(default=0.0, init=False)
    windows_evaluated: int = field(default=0, init=False)  # denominador da H3

    def __post_init__(self) -> None:
        self.window_size = max(1, int(round(self.fps * self.window_seconds)))
        self._window = deque(maxlen=self.window_size)

    def update(self, detections, capture_ts: float, now: float) -> WindowState:
        positive = any(
            d.cls_name == self.drowning_class and d.conf >= self.conf_threshold
            for d in detections
        )

        if positive and not self._window_has_positive():
            self._first_positive_ts = capture_ts  # marca o início do evento (H2)

        self._window.append(positive)

        window_full = len(self._window) == self.window_size
        ratio = sum(self._window) / len(self._window) if self._window else 0.0
        in_cooldown = (now - self._last_alert_ts) < self.cooldown_seconds

        if window_full:
            self.windows_evaluated += 1

        triggered = window_full and ratio >= self.trigger_ratio and not in_cooldown
        if triggered:
            self._last_alert_ts = now
            self._window.clear()  # reinicia o histórico após o disparo

        return WindowState(
            triggered=triggered,
            ratio=ratio,
            window_full=window_full,
            in_cooldown=in_cooldown,
            trigger_capture_ts=self._first_positive_ts,
        )

    def _window_has_positive(self) -> bool:
        return any(self._window)
