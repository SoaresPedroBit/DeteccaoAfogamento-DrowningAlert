"""Orquestração do sistema — integra as três camadas.

Uso:
  # Modo tempo real (câmera IP):
  python -m src.main --config config.yaml

  # Modo validação (cenários gravados, seção 8.6):
  python -m src.main --config config.yaml \
      --source videos/cenario_05.mp4 --scenario cenario_05_flutuando

  # Visualizar as detecções em janela (debug):
  python -m src.main --config config.yaml --show
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import yaml

from src.capture import VideoSource
from src.decision import DecisionEngine
from src.detector import Detector
from src.metrics import MetricsLogger
from src.notifier import SoundNotifier, TelegramNotifier


def _load_env_file(path: Path) -> None:
    """Carrega um .env local (gitignored) com credenciais, ex.: token do Telegram.

    Variáveis já definidas no ambiente têm precedência sobre o arquivo.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecção de afogamentos em piscinas")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None, help="sobrescreve source.uri (vídeo ou RTSP)")
    parser.add_argument("--scenario", default=None, help="nome do cenário p/ logs de métricas")
    parser.add_argument("--show", action="store_true", help="exibe janela com as detecções")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # .env ao lado do config.yaml (credenciais locais fora do versionamento)
    _load_env_file(Path(args.config).resolve().parent / ".env")

    uri = args.source or cfg["source"]["uri"]
    source = VideoSource(uri)
    fps = source.fps if not source.is_live else cfg["source"]["target_fps"]

    detector = Detector(
        weights=cfg["model"]["weights"],
        imgsz=cfg["model"]["imgsz"],
        device=cfg["model"]["device"],
        pool_class=cfg["model"]["pool_class"],
        roi_enabled=cfg["decision"]["roi_enabled"],
        roi_padding=cfg["decision"]["roi_padding"],
        roi_reset_frames=cfg["decision"].get("roi_reset_frames", 75),
    )

    engine = DecisionEngine(
        fps=fps,
        window_seconds=cfg["decision"]["window_seconds"],
        trigger_ratio=cfg["decision"]["trigger_ratio"],
        cooldown_seconds=cfg["decision"]["cooldown_seconds"],
        drowning_class=cfg["model"]["drowning_class"],
        conf_threshold=cfg["model"]["conf_threshold"],
    )

    notifier = SoundNotifier(cfg["notifier"]["sound_file"], cfg["notifier"]["repeat"])

    tg_cfg = cfg["notifier"].get("telegram", {})
    telegram = TelegramNotifier(
        bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", tg_cfg.get("bot_token", "")),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID", tg_cfg.get("chat_id", "")),
        send_snapshot=tg_cfg.get("send_snapshot", True),
    ) if tg_cfg.get("enabled", False) else None
    if tg_cfg.get("enabled", False) and (telegram is None or not telegram.enabled):
        print("[telegram] habilitado, mas token/chat_id ausentes — alertas não serão enviados.")

    metrics = MetricsLogger(
        cfg["metrics"]["output_dir"],
        args.scenario or cfg["metrics"].get("scenario", ""),
    ) if cfg["metrics"]["enabled"] else None

    frames_processed = 0
    print(f"[sistema] fonte: {uri} | fps: {fps:.0f} | "
          f"janela: {engine.window_size} frames | iniciando...")

    # Título ASCII: a HighGUI do Windows não renderiza UTF-8 na barra da janela.
    window_name = "Deteccao de afogamentos"
    display_scale = None  # calculado no primeiro frame para caber na tela

    try:
        while True:
            frame = source.read()
            if frame is None:
                if source.is_live:
                    time.sleep(0.005)
                    continue
                break  # fim do arquivo de vídeo

            detections = detector.predict(frame.image)
            state = engine.update(detections, frame.capture_ts, time.time())
            frames_processed += 1

            if state.triggered:
                emitted_ts = notifier.alert()
                latency = emitted_ts - state.trigger_capture_ts
                print(f"[ALERTA] afogamento detectado | razão da janela: "
                      f"{state.ratio:.0%} | latência: {latency:.2f}s")
                if telegram:
                    caption = (
                        "\U0001F6A8 Possível afogamento detectado\n"
                        f"Confiança da janela: {state.ratio:.0%}\n"
                        f"Latência: {latency:.2f}s\n"
                        f"Horário: {time.strftime('%d/%m/%Y %H:%M:%S')}"
                    )
                    scenario = args.scenario or cfg["metrics"].get("scenario", "")
                    if scenario:
                        caption += f"\nCenário: {scenario}"
                    telegram.alert(caption, frame.image.copy())
                if metrics:
                    metrics.log_alert(state.trigger_capture_ts, emitted_ts)

            if args.show:
                _draw(frame.image, detections, state)
                display = frame.image
                h, w = display.shape[:2]
                if display_scale is None:
                    display_scale = min(1280 / w, 720 / h, 1.0)
                if display_scale < 1.0:
                    # reduz só a cópia exibida; a inferência usa o frame original
                    display = cv2.resize(
                        display,
                        (int(w * display_scale), int(h * display_scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        source.release()
        cv2.destroyAllWindows()
        if metrics:
            metrics.finalize(engine.windows_evaluated, frames_processed)


def _draw(image, detections, state) -> None:
    # cores por nome de classe (BGR) — independente da ordem dos ids do modelo
    colors = {
        "piscinas": (255, 200, 0),
        "adulto_ok": (0, 200, 0),
        "crianca_ok": (0, 200, 200),
        "afogamento": (0, 0, 255),
    }
    for d in detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        color = colors.get(d.cls_name, (200, 200, 200))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{d.cls_name} {d.conf:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(image, f"janela: {state.ratio:.0%}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if state.ratio >= 0.6 else (255, 255, 255), 2)


if __name__ == "__main__":
    main()
