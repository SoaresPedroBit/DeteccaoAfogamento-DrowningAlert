"""Camada de notificação — alerta sonoro e mensagem via Telegram.

Tanto o som quanto o envio ao Telegram ocorrem em threads separadas para
não bloquear o laço de inferência: um alerta em curso não pode atrasar a
análise dos frames seguintes. No Windows o som usa `winsound` (stdlib, sem
dependências); em outras plataformas ou na ausência do arquivo WAV, recorre
ao beep do terminal. O Telegram usa a Bot API via HTTP (`requests`).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path


class SoundNotifier:
    def __init__(self, sound_file: str, repeat: int = 3):
        self.sound_file = Path(sound_file)
        self.repeat = repeat
        self._playing = threading.Event()

    def alert(self) -> float:
        """Dispara o alerta e retorna o timestamp de emissão (para a H2)."""
        emitted_ts = time.time()
        if not self._playing.is_set():
            threading.Thread(target=self._play, daemon=True).start()
        return emitted_ts

    def _play(self) -> None:
        self._playing.set()
        try:
            if self.sound_file.exists() and sys.platform == "win32":
                import winsound

                for _ in range(self.repeat):
                    winsound.PlaySound(str(self.sound_file), winsound.SND_FILENAME)
            else:
                for _ in range(self.repeat):
                    sys.stdout.write("\a")  # beep do terminal como fallback
                    sys.stdout.flush()
                    time.sleep(0.5)
        finally:
            self._playing.clear()


class TelegramNotifier:
    """Envia o alerta de afogamento para um chat/grupo via Telegram Bot API.

    O envio é feito em thread daemon para não bloquear a inferência. Quando
    há um frame disponível, anexa o snapshot do disparo (`sendPhoto`); caso
    contrário, ou se a codificação falhar, envia apenas texto (`sendMessage`).
    Fica inativo (`enabled=False`) se token ou chat_id não forem informados.
    """

    API_BASE = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        send_snapshot: bool = True,
        timeout: float = 10.0,
    ):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.send_snapshot = send_snapshot
        self.timeout = timeout
        self.enabled = bool(self.bot_token and self.chat_id)

    def alert(self, caption: str, image=None) -> None:
        """Dispara o envio de forma assíncrona (não bloqueia o chamador).

        `image` é um frame BGR (numpy); deve ser uma cópia, pois o laço
        principal pode reutilizar/desenhar sobre o buffer original.
        """
        if not self.enabled:
            return
        threading.Thread(
            target=self._send, args=(caption, image), daemon=True
        ).start()

    def _send(self, caption: str, image) -> None:
        try:
            import requests

            if self.send_snapshot and image is not None:
                photo = self._encode_jpeg(image)
                if photo is not None:
                    resp = requests.post(
                        f"{self.API_BASE}/bot{self.bot_token}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": caption},
                        files={"photo": ("alerta.jpg", photo, "image/jpeg")},
                        timeout=self.timeout,
                    )
                    self._check(resp)
                    return

            resp = requests.post(
                f"{self.API_BASE}/bot{self.bot_token}/sendMessage",
                data={"chat_id": self.chat_id, "text": caption},
                timeout=self.timeout,
            )
            self._check(resp)
        except Exception as exc:  # rede/token inválido não devem derrubar o sistema
            print(f"[telegram] falha ao enviar alerta: {exc}")

    @staticmethod
    def _encode_jpeg(image):
        try:
            import cv2

            ok, buffer = cv2.imencode(".jpg", image)
            return buffer.tobytes() if ok else None
        except Exception:
            return None

    @staticmethod
    def _check(resp) -> None:
        if resp.status_code != 200:
            print(f"[telegram] resposta {resp.status_code}: {resp.text[:200]}")
