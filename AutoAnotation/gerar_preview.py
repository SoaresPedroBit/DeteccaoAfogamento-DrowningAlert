"""
Gera imagens com as bboxes do YOLO desenhadas para revisao visual local
antes de subir no Roboflow.

Le images/ + labels/ do roboflow_import e gera preview/ com retangulos
coloridos e rotulo da classe em cada bbox.
"""
import argparse
import cv2
from collections import Counter
from pathlib import Path

CLASSES = {0: "adulto_ok", 1: "afogamento", 2: "crianca_ok", 3: "piscinas"}
# Cores BGR por classe (OpenCV usa BGR)
CORES = {
    0: (0, 255, 0),      # adulto_ok - verde
    1: (0, 0, 255),      # afogamento - vermelho
    2: (255, 255, 0),    # crianca_ok - ciano
    3: (255, 200, 0),    # piscinas - azul claro
}
ID_AFOGAMENTO = 1

ROOT_PADRAO = Path(__file__).parent / "output" / "roboflow_import"


def desenhar(img, labels):
    """Desenha bboxes na imagem. labels = lista de (cid, cx, cy, w, h) normalizados."""
    H, W = img.shape[:2]
    for cid, cx, cy, w, h in labels:
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        cor = CORES.get(cid, (200, 200, 200))
        espessura = 3 if cid == ID_AFOGAMENTO else 1  # afogamento mais grosso
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, espessura)
        # Etiqueta
        nome = CLASSES.get(cid, str(cid))
        (tw, th), _ = cv2.getTextSize(nome, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), cor, -1)
        cv2.putText(img, nome, (x1 + 2, max(th, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def ler_labels(label_path):
    out = []
    if not label_path.exists():
        return out
    for linha in label_path.read_text().splitlines():
        partes = linha.strip().split()
        if len(partes) != 5:
            continue
        try:
            cid = int(partes[0])
            cx, cy, w, h = map(float, partes[1:])
            out.append((cid, cx, cy, w, h))
        except ValueError:
            continue
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_PADRAO,
                        help=f"Pasta roboflow_import a usar (default: {ROOT_PADRAO})")
    parser.add_argument("--limite", type=int, default=0,
                        help="Limita N imagens (0 = todas)")
    parser.add_argument("--so-prioritarios", action="store_true",
                        help="So gera previews dos frames com classe afogamento")
    parser.add_argument("--so-baixa-conf", action="store_true",
                        help="Pula esta flag por enquanto (precisa do CSV)")
    args = parser.parse_args()

    images_dir = args.root / "images"
    labels_dir = args.root / "labels"
    preview_dir = args.root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        print(f"[ERRO] Pasta nao encontrada: {images_dir}")
        return

    imagens = sorted(images_dir.iterdir())
    if args.limite > 0:
        imagens = imagens[:args.limite]

    print(f"[INFO] Gerando preview para {len(imagens)} imagens em {preview_dir}")

    contagem_classes = Counter()
    gerados = 0
    pulados = 0

    for img_path in imagens:
        if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        labels = ler_labels(label_path)
        ids = {l[0] for l in labels}

        if args.so_prioritarios and ID_AFOGAMENTO not in ids:
            pulados += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [ERRO] nao leu {img_path.name}")
            continue
        img = desenhar(img, labels)
        cv2.imwrite(str(preview_dir / img_path.name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        for l in labels:
            contagem_classes[l[0]] += 1
        gerados += 1

    print(f"[OK] {gerados} previews gerados, {pulados} pulados")
    print(f"[OK] Pasta: {preview_dir}")
    print()
    print("Legenda das cores (BGR):")
    print("  piscinas    = azul claro    (linha fina)")
    print("  adulto_ok   = verde         (linha fina)")
    print("  crianca_ok  = ciano         (linha fina)")
    print("  afogamento  = vermelho      (linha GROSSA)")


if __name__ == "__main__":
    main()
