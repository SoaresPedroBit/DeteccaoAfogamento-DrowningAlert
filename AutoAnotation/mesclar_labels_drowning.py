"""
Pega as bboxes de 'drowning' (classe 2) do label ORIGINAL do dataset baixado
e injeta nos labels do nosso roboflow_import_falsos-neg/, remapeadas para a
nossa classe 'afogamento' (id 1).

Idempotente: nao duplica linhas se rodado mais de uma vez.

Uso:
    python mesclar_labels_drowning.py
    python mesclar_labels_drowning.py --target-dir <dir> \\
                                      --dataset-original <dir>
"""
import argparse
import csv
from pathlib import Path

DATASET_ORIGINAL_PADRAO = Path(
    "C:/Users/soare/Downloads/drowning-detection-dataset-main/"
    "drowning-detection-dataset-main/self-made dataset"
)
TARGET_PADRAO = (
    Path(__file__).parent / "output_drowning" / "roboflow_import_falsos-neg"
)


def ler_bboxes_classe(path, id_classe):
    """Retorna lista de tuplas (cx, cy, w, h) das bboxes da classe id_classe."""
    if not path.exists():
        return []
    out = []
    for linha in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        partes = linha.strip().split()
        if len(partes) != 5:
            continue
        try:
            cid = int(partes[0])
            if cid != id_classe:
                continue
            cx, cy, w, h = (float(x) for x in partes[1:])
            out.append((cx, cy, w, h))
        except ValueError:
            continue
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, default=TARGET_PADRAO,
                        help=f"Pasta com labels/ a serem mesclados (default: {TARGET_PADRAO})")
    parser.add_argument("--dataset-original", type=Path, default=DATASET_ORIGINAL_PADRAO,
                        help="Pasta raiz do dataset baixado (com images/ e labels/ por split)")
    parser.add_argument("--id-drowning-original", type=int, default=2,
                        help="ID da classe drowning no dataset original (default 2)")
    parser.add_argument("--id-afogamento-novo", type=int, default=1,
                        help="ID da classe afogamento no nosso esquema (default 1)")
    args = parser.parse_args()

    csv_path = args.target_dir / "relatorio.csv"
    labels_dir = args.target_dir / "labels"
    if not csv_path.is_file():
        print(f"[ERRO] CSV nao encontrado: {csv_path}")
        return
    if not labels_dir.is_dir():
        print(f"[ERRO] Pasta de labels nao encontrada: {labels_dir}")
        return

    n_total = 0
    n_atualizado = 0
    n_sem_bbox = 0
    n_skip_dupe = 0
    n_bboxes_injetadas = 0

    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_total += 1
            nome = row["nome_frame"]
            split = row.get("split_original") or ""
            # nome = "<split>_<id>" (ex: train_000123)
            if split and nome.startswith(split + "_"):
                id_orig = nome[len(split) + 1:]
            else:
                id_orig = nome

            lbl_orig = (
                args.dataset_original / "labels" / split / (id_orig + ".txt")
            ) if split else (args.dataset_original / "labels" / (id_orig + ".txt"))

            bboxes = ler_bboxes_classe(lbl_orig, args.id_drowning_original)
            if not bboxes:
                n_sem_bbox += 1
                continue

            # Le label nosso atual e ve o que ja tem
            lbl_nosso = labels_dir / (nome + ".txt")
            atual = lbl_nosso.read_text(encoding="utf-8") if lbl_nosso.exists() else ""
            linhas = atual.splitlines() if atual else []
            ja_existentes = set(linhas)

            adicionadas = 0
            for cx, cy, w, h in bboxes:
                nova = (
                    f"{args.id_afogamento_novo} "
                    f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
                if nova in ja_existentes:
                    n_skip_dupe += 1
                    continue
                linhas.append(nova)
                ja_existentes.add(nova)
                adicionadas += 1

            if adicionadas > 0:
                lbl_nosso.write_text("\n".join(linhas), encoding="utf-8")
                n_atualizado += 1
                n_bboxes_injetadas += adicionadas

    print("=" * 60)
    print(f"Frames no CSV:                {n_total}")
    print(f"Frames atualizados:           {n_atualizado}")
    print(f"Bboxes de afogamento injetadas: {n_bboxes_injetadas}")
    print(f"Frames sem drowning original (skip): {n_sem_bbox}")
    print(f"Linhas duplicadas (idempotencia):    {n_skip_dupe}")
    print("=" * 60)


if __name__ == "__main__":
    main()
