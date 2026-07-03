"""Registro de métricas experimentais — suporte às hipóteses H2 e H3.

H2 (latência): registra, para cada alerta, o intervalo entre o timestamp de
captura do primeiro frame positivo do evento e o timestamp de emissão do som.
O script de análise reporta média, desvio padrão e percentil 95 (seção 8.6).

H3 (falsos positivos): registra o total de janelas temporais avaliadas e o
total de alertas emitidos por cenário. Em cenários negativos, todo alerta é
falso positivo; a taxa é alertas / janelas avaliadas.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, output_dir: str, scenario: str = ""):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.scenario = scenario or "sem_cenario"
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.alerts_csv = self.dir / f"alertas_{self.scenario}_{ts}.csv"
        self.summary_csv = self.dir / f"resumo_{self.scenario}_{ts}.csv"
        self._alerts: list[dict] = []

    def log_alert(self, capture_ts: float, emitted_ts: float) -> None:
        self._alerts.append(
            {
                "cenario": self.scenario,
                "ts_captura": f"{capture_ts:.6f}",
                "ts_emissao": f"{emitted_ts:.6f}",
                "latencia_s": f"{emitted_ts - capture_ts:.3f}",
            }
        )

    def finalize(self, windows_evaluated: int, frames_processed: int) -> None:
        if self._alerts:
            with open(self.alerts_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._alerts[0].keys())
                writer.writeheader()
                writer.writerows(self._alerts)

        with open(self.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["cenario", "frames_processados", "janelas_avaliadas", "alertas_emitidos"]
            )
            writer.writerow(
                [self.scenario, frames_processed, windows_evaluated, len(self._alerts)]
            )

        print(f"[métricas] alertas: {len(self._alerts)} | "
              f"janelas: {windows_evaluated} | resumo: {self.summary_csv}")
