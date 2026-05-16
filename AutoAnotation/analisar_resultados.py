"""
Analise dos resultados do auto_anotate.py.
Le relatorio.csv e os .txt do roboflow_import/labels para gerar metricas detalhadas.
"""
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = {0: "adulto_ok", 1: "afogamento", 2: "crianca_ok", 3: "piscinas"}
ROOT = Path(__file__).parent / "output" / "roboflow_import"
CSV_PATH = ROOT / "relatorio.csv"
LABELS_DIR = ROOT / "labels"


def main():
    # ===== Frame-level: do CSV =====
    frames = []
    with CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frames.append({
                "nome": row["nome_frame"],
                "video": row["video"],
                "n_det": int(row["quantidade_deteccoes"]),
                "classes": row["classes_detectadas"].split(",") if row["classes_detectadas"] else [],
                "conf_media": float(row["confianca_media"]),
            })

    n_total = len(frames)
    n_com = sum(1 for f in frames if f["n_det"] > 0)
    n_sem = n_total - n_com

    # Frames por video
    por_video = Counter(f["video"] for f in frames)

    # Conjuntos de classes presentes em cada frame
    classes_por_frame = []
    for f in frames:
        cs = set(f["classes"]) - {""}
        classes_por_frame.append(cs)
    presenca = Counter()
    for cs in classes_por_frame:
        for c in cs:
            presenca[c] += 1

    # Total de deteccoes (multi-instance por frame conta cada)
    total_deteccoes = Counter()
    for f in frames:
        for c in f["classes"]:
            if c:
                total_deteccoes[c] += 1

    # Confianca media
    confs_com_det = [f["conf_media"] for f in frames if f["n_det"] > 0]
    media_conf = statistics.mean(confs_com_det) if confs_com_det else 0
    mediana_conf = statistics.median(confs_com_det) if confs_com_det else 0

    # Frames "limpos" (somente uma classe) vs "mistos"
    cont_tipo_frame = Counter()
    for cs in classes_por_frame:
        if not cs:
            cont_tipo_frame["sem_deteccao"] += 1
        elif len(cs) == 1:
            cont_tipo_frame[f"somente_{next(iter(cs))}"] += 1
        else:
            cont_tipo_frame["multiclasse"] += 1

    # ===== Per-detection: dos .txt =====
    # Cada linha do .txt: classe cx cy w h
    # Sem confianca por deteccao no formato YOLO, entao so podemos contar e medir bbox sizes.
    bbox_areas = defaultdict(list)
    for lbl in LABELS_DIR.glob("*.txt"):
        for linha in lbl.read_text().splitlines():
            partes = linha.strip().split()
            if len(partes) != 5:
                continue
            cid = int(partes[0])
            _cx, _cy, w, h = map(float, partes[1:])
            bbox_areas[cid].append(w * h)

    # ===== Saida =====
    p = print
    p("=" * 70)
    p("ANALISE DETALHADA - AUTO-ANOTACAO YOLO26N")
    p("=" * 70)
    p(f"Videos:                        {len(por_video)}")
    p(f"Frames extraidos:              {n_total}")
    p(f"Frames com deteccao:           {n_com}  ({100*n_com/n_total:.1f}%)")
    p(f"Frames sem deteccao:           {n_sem}")
    p(f"Media frames/video:            {n_total/len(por_video):.1f}")
    p(f"  min: {min(por_video.values())}, max: {max(por_video.values())}")
    p()
    p("CONFIANCA MEDIA POR FRAME (apenas frames com deteccao):")
    p(f"  media:   {media_conf:.3f}")
    p(f"  mediana: {mediana_conf:.3f}")
    p(f"  min:     {min(confs_com_det):.3f}")
    p(f"  max:     {max(confs_com_det):.3f}")
    p()
    p("TOTAL DE DETECCOES POR CLASSE (cada bbox conta):")
    soma = sum(total_deteccoes.values())
    for cid in sorted(CLASSES):
        n = total_deteccoes.get(CLASSES[cid], 0)
        pct = 100 * n / soma if soma else 0
        p(f"  [{cid}] {CLASSES[cid]:<12} {n:>6}  ({pct:5.1f}%)")
    p(f"  {'TOTAL':<16} {soma:>6}")
    p()
    p("PRESENCA POR FRAME (% de frames em que a classe aparece pelo menos 1x):")
    for cid in sorted(CLASSES):
        n = presenca.get(CLASSES[cid], 0)
        p(f"  [{cid}] {CLASSES[cid]:<12} {n:>5}/{n_total}  ({100*n/n_total:5.1f}%)")
    p()
    p("COMPOSICAO DOS FRAMES:")
    for tipo, n in cont_tipo_frame.most_common():
        p(f"  {tipo:<25} {n:>5}  ({100*n/n_total:5.1f}%)")
    p()
    p("AREA MEDIA DE BBOX POR CLASSE (% da imagem):")
    for cid in sorted(CLASSES):
        areas = bbox_areas.get(cid, [])
        if areas:
            med = statistics.mean(areas) * 100
            md = statistics.median(areas) * 100
            p(f"  [{cid}] {CLASSES[cid]:<12} media={med:5.2f}%  mediana={md:5.2f}%  n={len(areas)}")
        else:
            p(f"  [{cid}] {CLASSES[cid]:<12} (sem deteccoes)")
    p()
    p("INDICADORES DE QUALIDADE:")
    # Sinais de bias do modelo
    cls3_pct = 100 * presenca.get("afogamento", 0) / n_total
    cls1_pct = 100 * presenca.get("adulto_ok", 0) / n_total
    p(f"  Cobertura geral:                 {100*n_com/n_total:.1f}% (alto = bom)")
    p(f"  % frames com afogamento:         {cls3_pct:.1f}%")
    p(f"  % frames com adulto_ok:          {cls1_pct:.1f}%")
    p(f"  Ratio crianca_ok / adulto_ok:    {total_deteccoes.get('crianca_ok',0) / max(1,total_deteccoes.get('adulto_ok',0)):.1f}x")
    if cls3_pct > 80:
        p("  >> ALERTA: classe afogamento aparece em >80% dos frames -")
        p("     pode ser bias do modelo. Revisar manualmente uma amostra.")
    if cls1_pct < 5:
        p("  >> ALERTA: classe adulto_ok aparece em <5% dos frames -")
        p("     modelo provavelmente confunde adultos com criancas (ou playlist so tem criancas).")
    p()
    p("TOP 5 FRAMES COM MAIOR NUMERO DE DETECCOES (candidatos a confusao):")
    for f in sorted(frames, key=lambda x: x["n_det"], reverse=True)[:5]:
        p(f"  {f['n_det']:>3} dets, conf={f['conf_media']:.3f}  ->  {f['nome']}")
    p()
    p("TOP 5 FRAMES COM MENOR CONFIANCA (candidatos a falsos positivos):")
    for f in sorted([x for x in frames if x["n_det"] > 0], key=lambda x: x["conf_media"])[:5]:
        p(f"  conf={f['conf_media']:.3f}, dets={f['n_det']}  ->  {f['nome']}")
    p()
    p("FRAMES SEM DETECCAO ALGUMA (revisar para entender):")
    sem = [f for f in frames if f["n_det"] == 0]
    for f in sem:
        p(f"  {f['nome']}  ({f['video']})")
    p()
    p("FRAMES COM CLASSE adulto_ok (raros - revisar):")
    com_adulto = [f for f in frames if "adulto_ok" in f["classes"]]
    p(f"  Total: {len(com_adulto)} frames")
    for f in com_adulto[:10]:
        p(f"    {f['nome']}  classes={f['classes']}  conf={f['conf_media']:.3f}")
    if len(com_adulto) > 10:
        p(f"    ... (+{len(com_adulto)-10} outros)")
    p("=" * 70)


if __name__ == "__main__":
    main()
