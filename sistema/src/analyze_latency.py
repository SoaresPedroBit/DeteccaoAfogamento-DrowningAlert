"""Análise das latências registradas (hipótese H2).

Consolida todos os CSVs de alertas em logs/ e reporta média, desvio padrão
e percentil 95 da latência, conforme protocolo da seção 8.6 do TCC.

Uso:
  python -m src.analyze_latency --logs logs/
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    args = parser.parse_args()

    latencies: list[float] = []
    for path in sorted(Path(args.logs).glob("alertas_*.csv")):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                latencies.append(float(row["latencia_s"]))

    if not latencies:
        print("Nenhum alerta registrado em", args.logs)
        return

    latencies.sort()
    n = len(latencies)
    p95 = latencies[min(n - 1, int(round(0.95 * (n - 1))))]

    print(f"Execuções (alertas): {n}")
    print(f"Média:               {statistics.mean(latencies):.3f} s")
    print(f"Desvio padrão:       {statistics.stdev(latencies) if n > 1 else 0:.3f} s")
    print(f"Percentil 95:        {p95:.3f} s")
    print(f"H2 (média < 10 s):   {'ATENDIDA' if statistics.mean(latencies) < 10 else 'NÃO ATENDIDA'}")
    print(f"H2 no P95 (< 10 s):  {'ATENDIDA' if p95 < 10 else 'NÃO ATENDIDA'}")


if __name__ == "__main__":
    main()
