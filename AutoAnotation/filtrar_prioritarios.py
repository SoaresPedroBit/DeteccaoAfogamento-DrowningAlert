"""
Filtra um roboflow_import/ ja gerado pelo anotar_imagens.py, copiando para
uma pasta nova apenas os frames marcados como prioritarios na revisao.

Caso de uso: o output_drowning/roboflow_import/ tem 8572 frames, mas pra
balancear o dataset de treino voce so quer subir no Roboflow os 1909 que
tem 'drowning' no label original (nao diluir as outras classes).

Modos:
  --modo prioritarios   Todos os frames com drowning original (default, ~1909)
  --modo falsos-neg     So frames onde nosso modelo NAO detectou afogamento
                        mas o original tem drowning (~481, mais valiosos)
  --modo verdadeiros    So onde nosso modelo TAMBEM detectou afogamento
                        (~1428, revisao mais leve)

Uso:
    python filtrar_prioritarios.py
    python filtrar_prioritarios.py --modo falsos-neg
    python filtrar_prioritarios.py --root <outra/roboflow_import> \\
                                   --output <destino>
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT_PADRAO = Path(__file__).parent / "output_drowning" / "roboflow_import"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_PADRAO,
                        help=f"Pasta roboflow_import de origem (default: {ROOT_PADRAO})")
    parser.add_argument("--output", type=Path, default=None,
                        help="Pasta de destino. Default: <root_pai>/roboflow_import_<modo>/")
    parser.add_argument("--modo", choices=["prioritarios", "falsos-neg", "verdadeiros"],
                        default="prioritarios",
                        help="Que subconjunto filtrar (default: prioritarios)")
    parser.add_argument("--copiar-preview", action="store_true",
                        help="Tambem copia imagens da pasta preview/ se existir")
    args = parser.parse_args()

    csv_path = args.root / "relatorio.csv"
    if not csv_path.is_file():
        print(f"[ERRO] relatorio.csv nao encontrado em {args.root}")
        sys.exit(1)

    if args.output is None:
        args.output = args.root.parent / f"roboflow_import_{args.modo}"

    images_src = args.root / "images"
    labels_src = args.root / "labels"
    preview_src = args.root / "preview"
    images_dst = args.output / "images"
    labels_dst = args.output / "labels"
    preview_dst = args.output / "preview"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)
    if args.copiar_preview and preview_src.is_dir():
        preview_dst.mkdir(parents=True, exist_ok=True)

    # Le e filtra
    todos = []
    selecionados = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            todos.append(row)
            tem_orig = row.get("tem_drowning_original") == "sim"
            modelo_pegou = row.get("modelo_pegou_afogamento") == "sim"
            if args.modo == "prioritarios" and tem_orig:
                selecionados.append(row)
            elif args.modo == "falsos-neg" and tem_orig and not modelo_pegou:
                selecionados.append(row)
            elif args.modo == "verdadeiros" and tem_orig and modelo_pegou:
                selecionados.append(row)

    if not selecionados:
        print(f"[AVISO] Nenhum frame casa com modo={args.modo}.")
        print(f"        Total no CSV: {len(todos)}.")
        print(f"        Verifique se o CSV tem as colunas tem_drowning_original e modelo_pegou_afogamento.")
        sys.exit(1)

    print(f"[INFO] Total no CSV:  {len(todos)}")
    print(f"[INFO] Selecionados ({args.modo}): {len(selecionados)}")
    print(f"[INFO] Destino: {args.output}")

    # Copia imagens, labels e (opcional) previews
    n_img = n_lbl = n_prev = 0
    n_img_falta = n_lbl_falta = 0
    for r in selecionados:
        nome = r["nome_frame"]
        img_in = images_src / (nome + ".jpg")
        lbl_in = labels_src / (nome + ".txt")
        img_out = images_dst / (nome + ".jpg")
        lbl_out = labels_dst / (nome + ".txt")
        if img_in.is_file():
            if not img_out.exists():
                shutil.copy2(img_in, img_out)
            n_img += 1
        else:
            n_img_falta += 1
        if lbl_in.is_file():
            if not lbl_out.exists():
                shutil.copy2(lbl_in, lbl_out)
            n_lbl += 1
        else:
            n_lbl_falta += 1
        if args.copiar_preview and preview_src.is_dir():
            prev_in = preview_src / (nome + ".jpg")
            prev_out = preview_dst / (nome + ".jpg")
            if prev_in.is_file() and not prev_out.exists():
                shutil.copy2(prev_in, prev_out)
                n_prev += 1

    # classes.txt: copia direto
    classes_txt_in = args.root / "classes.txt"
    if classes_txt_in.is_file():
        shutil.copy2(classes_txt_in, args.output / "classes.txt")

    # CSV filtrado, mesmas colunas do original
    csv_out = args.output / "relatorio.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        if selecionados:
            w = csv.DictWriter(f, fieldnames=list(selecionados[0].keys()))
            w.writeheader()
            w.writerows(selecionados)

    # prioritarios_revisar.txt: lista todos os selecionados (ja sao todos prioritarios)
    prio_path = args.output / "prioritarios_revisar.txt"
    with prio_path.open("w", encoding="utf-8") as f:
        f.write(f"# Subconjunto filtrado de {args.root.name} (modo={args.modo})\n")
        f.write(f"# Total: {len(selecionados)}\n\n")
        for r in selecionados:
            f.write(r["nome_frame"] + "\n")

    # Estatisticas das classes preditas no subconjunto
    contagem = Counter()
    for r in selecionados:
        for c in (r.get("classes_preditas") or "").split(","):
            c = c.strip()
            if c:
                contagem[c] += 1

    print()
    print("=" * 60)
    print(f"[OK] {n_img} imagens copiadas (faltaram {n_img_falta})")
    print(f"[OK] {n_lbl} labels copiados (faltaram {n_lbl_falta})")
    if args.copiar_preview:
        print(f"[OK] {n_prev} previews copiados")
    print(f"[OK] CSV filtrado -> {csv_out}")
    print(f"[OK] Lista        -> {prio_path}")
    print()
    print("Distribuicao de deteccoes do nosso modelo no subconjunto:")
    for nome, n in sorted(contagem.items(), key=lambda x: -x[1]):
        print(f"  {nome:<14} {n:>5}")
    print("=" * 60)


if __name__ == "__main__":
    main()
