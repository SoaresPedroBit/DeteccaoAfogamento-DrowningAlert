"""Camada de processamento (parte 1) — inferência YOLO sobre cada frame.

Encapsula o modelo Ultralytics e aplica a ROI derivada da classe `piscinas`:
quando a piscina é detectada, os frames seguintes são analisados
prioritariamente dentro dessa região (com padding), reduzindo o custo
computacional de áreas irrelevantes (seção 8.5 do TCC).

O mapeamento `id -> nome` vem de `model.names` em runtime (fonte da verdade),
evitando drift entre constantes hardcoded e a ordem aprendida pelo modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # coordenadas no frame original


@dataclass
class Detector:
    weights: str
    imgsz: int = 640
    device: str = "cuda"
    pool_class: str = "piscinas"
    roi_enabled: bool = True
    roi_padding: float = 0.10
    roi_reset_frames: int = 75  # frames sem piscina na ROI até resetá-la (~3 s a 25 FPS)

    def __post_init__(self) -> None:
        self.model = YOLO(self.weights)
        # model.names: {id: nome} aprendido no treino — fonte da verdade
        self.id_to_name: dict[int, str] = dict(self.model.names)
        self.name_to_id: dict[str, int] = {v: k for k, v in self.id_to_name.items()}
        if self.pool_class not in self.name_to_id:
            raise ValueError(
                f"Classe '{self.pool_class}' não existe no modelo. "
                f"Classes disponíveis: {sorted(self.name_to_id)}"
            )
        self.pool_cls = self.name_to_id[self.pool_class]
        self._roi: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2
        self._roi_misses = 0  # frames consecutivos sem piscina dentro da ROI

    # ------------------------------------------------------------------ infer
    def predict(self, image: np.ndarray) -> list[Detection]:
        crop, offset = self._apply_roi(image)

        results = self.model.predict(
            crop, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]

        detections: list[Detection] = []
        ox, oy = offset
        for box in results.boxes:
            cls_id = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=self.id_to_name.get(cls_id, str(cls_id)),
                    conf=float(box.conf),
                    xyxy=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                )
            )

        self._update_roi(image, detections)
        return detections

    # -------------------------------------------------------------------- ROI
    def _apply_roi(self, image: np.ndarray):
        if not self.roi_enabled or self._roi is None:
            return image, (0, 0)
        x1, y1, x2, y2 = self._roi
        return image[y1:y2, x1:x2], (x1, y1)

    def _update_roi(self, image: np.ndarray, detections: list[Detection]) -> None:
        pools = [d for d in detections if d.cls_id == self.pool_cls]
        if not pools:
            # Sem piscina no recorte atual: após N frames consecutivos,
            # volta ao frame inteiro (câmera pode ter se movido ou a ROI
            # inicial pode estar errada).
            if self._roi is not None:
                self._roi_misses += 1
                if self._roi_misses >= self.roi_reset_frames:
                    self.reset_roi()
            return

        self._roi_misses = 0
        best = max(pools, key=lambda d: d.conf)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = best.xyxy
        px = (x2 - x1) * self.roi_padding
        py = (y2 - y1) * self.roi_padding
        self._roi = (
            max(0, int(x1 - px)),
            max(0, int(y1 - py)),
            min(w, int(x2 + px)),
            min(h, int(y2 + py)),
        )

    def reset_roi(self) -> None:
        self._roi = None
        self._roi_misses = 0
