"""Popula o dashboard com dados sintéticos para desenvolvimento e demonstração.

Gera eventos (com snapshots), amostras de desempenho dos dois modelos,
`latest.jpg` e `status.json` — permitindo exercitar as três telas sem o
processo de inferência rodando. NÃO usar com o banco de produção: o script
recusa executar se já houver eventos reais, a menos que receba --forcar.

Uso:
  conda activate tcc
  python gerar_dados_demo.py [--dias 14] [--forcar]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timedelta

import cv2
import numpy as np

import db


def frame_sintetico(texto: str, alerta: bool = False) -> np.ndarray:
    img = np.full((720, 1280, 3), (146, 109, 36), dtype=np.uint8)  # água (BGR)
    ruido = np.random.randint(0, 28, img.shape, dtype=np.uint8)
    img = cv2.add(img, ruido)
    cv2.rectangle(img, (90, 120), (1190, 660), (196, 160, 60), 3)
    cv2.putText(img, "piscinas 0.91", (95, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (196, 160, 60), 2)
    cor = (0, 0, 255) if alerta else (0, 200, 0)
    rotulo = "afogamento 0.78" if alerta else "adulto_ok 0.84"
    cv2.rectangle(img, (560, 330), (700, 520), cor, 2)
    cv2.putText(img, rotulo, (560, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    cv2.putText(img, texto, (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dias", type=int, default=14)
    parser.add_argument("--forcar", action="store_true")
    args = parser.parse_args()

    db.init()
    with db.connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM eventos").fetchone()["c"]
    if total and not args.forcar:
        raise SystemExit(
            f"O banco já tem {total} evento(s). Use --forcar apenas se tiver certeza "
            "de que são dados de teste."
        )

    rng = random.Random(42)
    agora = datetime.now()
    modelos = ["yolo11n", "yolo26n"]

    # ------------------------------------------------------------- eventos
    eventos = 0
    with db.connect() as conn:
        for dia in range(args.dias):
            data = agora - timedelta(days=args.dias - 1 - dia)
            # mais eventos nos horários de uso da piscina (10h-18h)
            for _ in range(rng.randint(1, 5)):
                hora = min(23, max(0, int(rng.gauss(14, 3))))
                ts_alerta = data.replace(hour=hora, minute=rng.randint(0, 59),
                                         second=rng.randint(0, 59), microsecond=0)
                if ts_alerta > agora:
                    continue
                latencia_ms = int(rng.gauss(4200, 900))
                latencia_ms = max(3000, min(8000, latencia_ms))
                ts_inicio = ts_alerta - timedelta(milliseconds=latencia_ms)
                falso = rng.random() < 0.25
                conf_media = round(rng.uniform(0.32, 0.55) if falso
                                   else rng.uniform(0.5, 0.88), 4)
                pendente = rng.random() < 0.2
                status = "pendente" if pendente else ("falso_positivo" if falso else "confirmado")
                frames_janela = 75
                frames_pos = rng.randint(38, 74)
                modelo = rng.choice(modelos)

                cur = conn.execute(
                    "INSERT INTO eventos (timestamp_inicio, timestamp_alerta, camera_id, "
                    "classe, confianca_media, confianca_maxima, frames_positivos, "
                    "frames_janela, latencia_ms, modelo, caminho_snapshot, status, revisado_em) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?)",
                    (ts_inicio.isoformat(timespec="milliseconds"),
                     ts_alerta.isoformat(timespec="milliseconds"),
                     "cam01", "afogamento", conf_media,
                     round(min(0.99, conf_media + rng.uniform(0.05, 0.2)), 4),
                     frames_pos, frames_janela, latencia_ms, modelo,
                     status,
                     None if pendente else ts_alerta.isoformat(timespec="seconds")),
                )
                evento_id = cur.lastrowid
                pasta = db.SNAPSHOTS_DIR / f"{ts_alerta:%Y}" / f"{ts_alerta:%m}"
                pasta.mkdir(parents=True, exist_ok=True)
                img = frame_sintetico(f"DEMO evento #{evento_id} {ts_alerta:%d/%m %H:%M}", alerta=True)
                cv2.imwrite(str(pasta / f"{evento_id}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                conn.execute("UPDATE eventos SET caminho_snapshot=? WHERE id=?",
                             (f"snapshots/{ts_alerta:%Y}/{ts_alerta:%m}/{evento_id}.jpg", evento_id))
                eventos += 1

        # ------------------------------------------- amostras de desempenho
        amostras = 0
        for modelo, fps_base, inf_base in (("yolo11n", 24.0, 11.5), ("yolo26n", 21.5, 14.0)):
            for dia in range(args.dias):
                data = agora - timedelta(days=args.dias - 1 - dia)
                for hora in range(9, 19):
                    for minuto in (0, 20, 40):
                        ts = data.replace(hour=hora, minute=minuto, second=0, microsecond=0)
                        if ts > agora:
                            continue
                        conn.execute(
                            "INSERT INTO metricas_desempenho "
                            "(timestamp, modelo, fps, inferencia_ms_media, inferencia_ms_p95, frames) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (ts.isoformat(timespec="seconds"), modelo,
                             round(rng.gauss(fps_base, 1.2), 2),
                             round(rng.gauss(inf_base, 1.0), 2),
                             round(rng.gauss(inf_base + 4, 1.4), 2),
                             int(fps_base * 300)),
                        )
                        amostras += 1

    # ------------------------------------------------- ao vivo (stream demo)
    img = frame_sintetico(f"DEMO ao vivo {agora:%d/%m/%Y %H:%M}")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    (db.DATA_DIR / "latest.jpg").write_bytes(buf.tobytes())
    (db.DATA_DIR / "status.json").write_text(json.dumps({
        "camera_id": "cam01", "modelo": "yolo11n", "fonte": "demo",
        "fps": 23.7, "inferencia_ms_media": 11.8,
        "iniciado_em": time.time() - 3600, "ultimo_frame_ts": time.time(),
    }), encoding="utf-8")

    print(f"[demo] {eventos} eventos e {amostras} amostras de desempenho gerados "
          f"em {db.DB_PATH}")
    print("[demo] observação: status.json aponta 'agora' — a tela Ao vivo mostra "
          "a câmera como conectada por ~5s e depois indica desconexão (esperado).")


if __name__ == "__main__":
    main()
