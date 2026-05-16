"""
Re-anotacao de um dataset YOLO externo com nosso modelo treinado.

Caso de uso: voce tem um dataset com imagens + labels YOLO (em outro conjunto
de classes) e quer:
  1. Gerar novas anotacoes no nosso esquema de 4 classes
     (0=adulto_ok, 1=afogamento, 2=crianca_ok, 3=piscinas).
  2. Marcar como prioritarios os frames cujo label ORIGINAL tem a classe
     informada em --id-drowning-original (default 2 = drowning no dataset
     "drowning-detection-dataset-main"), assim a revisao manual no Roboflow
     comeca pelos casos mais valiosos.

Saida: layout pronto para import no Roboflow em <output>/roboflow_import/.

Uso:
    python anotar_imagens.py --dataset-dir <pasta_com_images_e_labels> \\
                              --model <best.pt> \\
                              [--output AutoAnotation/output_drowning] \\
                              [--imgsz 640] [--conf 0.25] \\
                              [--id-drowning-original 2]
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2


def redimensionar(img, imgsz):
    """Reduz a imagem para caber em imgsz x imgsz mantendo a proporcao. Nao aumenta."""
    h, w = img.shape[:2]
    if max(h, w) <= imgsz:
        return img
    if w >= h:
        novo_w, novo_h = imgsz, max(1, int(h * imgsz / w))
    else:
        novo_h, novo_w = imgsz, max(1, int(w * imgsz / h))
    return cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_AREA)


def listar_imagens(dataset_dir):
    """Coleta (split, image_path, original_label_path) para imagens em images/{train,val}/."""
    imagens_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"
    if not imagens_root.is_dir():
        # Fallback: dataset com pasta unica (sem splits)
        return [
            ("", p, labels_root / (p.stem + ".txt"))
            for p in sorted(dataset_dir.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

    items = []
    for split_dir in sorted(imagens_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        labels_split = labels_root / split
        for img in sorted(split_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            items.append((split, img, labels_split / (img.stem + ".txt")))
    return items


def ler_classes_originais(label_path):
    """Le os IDs de classe do label original. Retorna set vazio se nao existir."""
    if not label_path.exists():
        return set()
    ids = set()
    for linha in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        partes = linha.strip().split()
        if len(partes) >= 1:
            try:
                ids.add(int(partes[0]))
            except ValueError:
                pass
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Re-anotacao de dataset YOLO externo com nosso modelo"
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="Pasta raiz com images/ e labels/ no formato YOLO")
    parser.add_argument("--model", required=True, help="Caminho para o best.pt")
    parser.add_argument("--output", default="./output_drowning",
                        help="Pasta de saida (default: ./output_drowning)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Lado maximo das imagens copiadas para o Roboflow (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confianca minima da deteccao (default: 0.25)")
    parser.add_argument("--id-drowning-original", type=int, default=2,
                        help="ID da classe de afogamento no dataset ORIGINAL "
                             "(default 2 = drowning no drowning-detection-dataset)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_path = Path(args.model).resolve()
    if not dataset_dir.is_dir():
        print(f"[ERRO] Dataset nao encontrado: {dataset_dir}")
        sys.exit(1)
    if not model_path.is_file():
        print(f"[ERRO] Modelo nao encontrado: {model_path}")
        sys.exit(1)

    output = Path(args.output).resolve()
    roboflow_dir = output / "roboflow_import"
    out_images = roboflow_dir / "images"
    out_labels = roboflow_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RE-ANOTACAO YOLO (dataset externo -> Roboflow)")
    print("=" * 60)
    print(f"Dataset:  {dataset_dir}")
    print(f"Modelo:   {model_path}")
    print(f"Saida:    {roboflow_dir}")
    print(f"imgsz:    {args.imgsz}")
    print(f"conf:     {args.conf}")
    print(f"drowning original id: {args.id_drowning_original}")
    print("=" * 60)

    print("\n[INFO] Coletando imagens...")
    itens = listar_imagens(dataset_dir)
    if not itens:
        print("[ERRO] Nenhuma imagem encontrada.")
        sys.exit(1)
    print(f"[OK] {len(itens)} imagens encontradas")

    print(f"\n[INFO] Carregando modelo: {model_path}")
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    nomes_por_id = {int(k): v for k, v in model.names.items()}
    print(f"[OK] Modelo carregado. Classes:")
    for cid in sorted(nomes_por_id):
        print(f"     [{cid}] {nomes_por_id[cid]}")
    id_afogamento = next(
        (cid for cid, n in nomes_por_id.items() if n == "afogamento"), None,
    )

    try:
        from tqdm import tqdm
        iterador = tqdm(itens, desc="anotando", unit="img")
    except ImportError:
        iterador = itens
        print("[AVISO] tqdm nao instalado, sem barra de progresso")

    registros = []
    contagem_nova = Counter()
    n_drowning_original = 0
    n_modelo_pegou_afogamento = 0

    for split, img_path, lbl_orig_path in iterador:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] nao leu {img_path}")
            continue

        # Inferencia na imagem ORIGINAL (YOLO faz letterbox internamente).
        # Bboxes normalizados sao independentes da resolucao final.
        res = model.predict(source=img, conf=args.conf, verbose=False)[0]

        novo_nome = f"{split}_{img_path.stem}" if split else img_path.stem
        out_img_path = out_images / (novo_nome + ".jpg")
        out_lbl_path = out_labels / (novo_nome + ".txt")

        # Salva imagem redimensionada para imgsz (para Roboflow)
        if not out_img_path.exists():
            img_resized = redimensionar(img, args.imgsz)
            cv2.imwrite(str(out_img_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Escreve label novo (sempre, mesmo se vazio, para o Roboflow)
        linhas = []
        classes_frame = []
        confs_frame = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for c, (cx, cy, w, h), cf in zip(cls, xywhn, confs):
                linhas.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                classes_frame.append(int(c))
                confs_frame.append(float(cf))
        out_lbl_path.write_text("\n".join(linhas), encoding="utf-8")

        # Comparacao com label original
        classes_originais = ler_classes_originais(lbl_orig_path)
        tem_drowning_original = args.id_drowning_original in classes_originais
        modelo_pegou_afogamento = (id_afogamento is not None
                                    and id_afogamento in classes_frame)
        if tem_drowning_original:
            n_drowning_original += 1
            if modelo_pegou_afogamento:
                n_modelo_pegou_afogamento += 1

        for c in classes_frame:
            contagem_nova[c] += 1

        registros.append({
            "nome": novo_nome,
            "split_original": split,
            "n_deteccoes": len(classes_frame),
            "classes_predito": classes_frame,
            "conf_media": (sum(confs_frame) / len(confs_frame)) if confs_frame else 0.0,
            "classes_originais": sorted(classes_originais),
            "tem_drowning_original": tem_drowning_original,
            "modelo_pegou_afogamento": modelo_pegou_afogamento,
        })

    # classes.txt na ordem do modelo
    classes_txt = roboflow_dir / "classes.txt"
    classes_txt.write_text(
        "\n".join(nomes_por_id[i] for i in sorted(nomes_por_id)) + "\n",
        encoding="utf-8",
    )
    print(f"\n[OK] classes.txt -> {classes_txt}")

    # relatorio.csv
    csv_path = roboflow_dir / "relatorio.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "nome_frame", "split_original", "quantidade_deteccoes",
            "classes_preditas", "confianca_media",
            "classes_originais", "tem_drowning_original",
            "modelo_pegou_afogamento",
        ])
        for r in registros:
            classes_nomes = ",".join(
                nomes_por_id.get(c, str(c)) for c in r["classes_predito"]
            )
            w.writerow([
                r["nome"], r["split_original"], r["n_deteccoes"],
                classes_nomes, f"{r['conf_media']:.4f}",
                ",".join(str(c) for c in r["classes_originais"]),
                "sim" if r["tem_drowning_original"] else "nao",
                "sim" if r["modelo_pegou_afogamento"] else "nao",
            ])
    print(f"[OK] relatorio.csv -> {csv_path}")

    # prioritarios_revisar.txt = todos os frames com drowning original
    prio_path = roboflow_dir / "prioritarios_revisar.txt"
    prio = [r for r in registros if r["tem_drowning_original"]]
    with prio_path.open("w", encoding="utf-8") as f:
        f.write(f"# Frames com 'drowning' (classe {args.id_drowning_original}) "
                f"no label ORIGINAL - revisar primeiro no Roboflow\n")
        f.write(f"# Total: {len(prio)}\n\n")
        for r in prio:
            f.write(r["nome"] + "\n")
    print(f"[OK] prioritarios_revisar.txt -> {prio_path} ({len(prio)} frames)")

    # Resumo + validacao recall do modelo na classe afogamento
    recall_pct = (100 * n_modelo_pegou_afogamento / n_drowning_original
                  if n_drowning_original else 0.0)
    print("\n" + "=" * 60)
    print("RELATORIO FINAL")
    print("=" * 60)
    print(f"Imagens processadas:       {len(registros)}")
    print()
    print("Deteccoes do nosso modelo por classe:")
    for cid in sorted(nomes_por_id):
        print(f"  [{cid}] {nomes_por_id[cid]:<14} {contagem_nova.get(cid, 0):>6}")
    print()
    print(f"Frames com drowning original:        {n_drowning_original}")
    print(f"Modelo tambem detectou afogamento:   {n_modelo_pegou_afogamento}  "
          f"(frame-level recall {recall_pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
