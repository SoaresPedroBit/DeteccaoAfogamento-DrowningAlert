"""
Pipeline de pre-anotacao automatica para deteccao de afogamento.

Processa uma playlist do YouTube e gera pre-anotacoes YOLO prontas para
revisao no Roboflow.

Etapas:
  1. Download da playlist (yt-dlp)
  2. Extracao de frames a cada N segundos (ffmpeg)
  3. Inferencia com modelo YOLO treinado
  4. Organizacao no layout do Roboflow (images/ + labels/)
  5. Relatorio CSV e lista de prioritarios (classe 3 = afogamento)

Uso:
    python auto_anotate.py --playlist <URL> --model <best.pt>
    python auto_anotate.py --playlist <URL> --model <best.pt> --fps 5 --conf 0.3

Dependencias:
    pip install -r requirements.txt
    + ffmpeg no PATH do sistema
"""
import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

# Ordem real aprendida pelo modelo (ver datasets/.../data.yaml).
# Em runtime usamos model.names para evitar drift, mas mantemos este dict
# como fallback documentado.
CLASSES = {
    0: "adulto_ok",
    1: "afogamento",
    2: "crianca_ok",
    3: "piscinas",
}
CLASSE_PRIORITARIA_NOME = "afogamento"


def sanitize(nome):
    """Remove caracteres invalidos para nome de arquivo/pasta."""
    nome = re.sub(r"[^\w\-_. ]", "_", nome)
    nome = re.sub(r"\s+", "_", nome).strip("_")
    return nome[:80]


# Padrao de stream do yt-dlp: "<base>.fNNN.<ext>"
PADRAO_STREAM = re.compile(r"^(.+)\.f\d+$")


def nome_logico(video_path):
    """Remove sufixo .fNNN do stem do yt-dlp para obter o nome 'logico' do video."""
    stem = video_path.stem
    m = PADRAO_STREAM.match(stem)
    return m.group(1) if m else stem


def selecionar_videos(videos_dir):
    """Dedup por nome logico e mantem apenas o arquivo com stream de video.

    Quando o yt-dlp falha em mesclar, sobram pares (.fNNN.webm video + .fNNN.webm audio).
    cv2.VideoCapture abre ambos, mas o de audio retorna 0 frames. Filtramos pelo width>0.
    """
    candidatos = defaultdict(list)
    for p in videos_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in {".mp4", ".mkv", ".webm", ".m4a"}:
            continue
        candidatos[nome_logico(p)].append(p)

    escolhidos = []
    for logico, arquivos in sorted(candidatos.items()):
        # Tenta abrir cada um e fica com o que tiver stream de video
        for arq in arquivos:
            cap = cv2.VideoCapture(str(arq))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                escolhidos.append((logico, arq))
                break
        else:
            print(f"  [SKIP] {logico}: nenhum arquivo com stream de video")
    return escolhidos


# ============================================================
# Etapa 1: Download da playlist
# ============================================================
def baixar_playlist(playlist_url, videos_dir):
    """Baixa todos os videos da playlist em 720p (ou melhor disponivel)."""
    import yt_dlp

    videos_dir.mkdir(parents=True, exist_ok=True)
    archive_file = videos_dir / ".download_archive.txt"

    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": str(videos_dir / "%(playlist_index)03d_%(title).80s.%(ext)s"),
        "merge_output_format": "mp4",
        "ignoreerrors": True,
        "download_archive": str(archive_file),
        "continuedl": True,
        "nooverwrites": True,
        "quiet": False,
        "no_warnings": False,
    }

    print(f"\n[INFO] Baixando playlist: {playlist_url}")
    print(f"[INFO] Destino: {videos_dir}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([playlist_url])
    except Exception as e:
        print(f"[AVISO] Erro no download da playlist (continuando com o que ja foi baixado): {e}")

    escolhidos = selecionar_videos(videos_dir)
    print(f"[OK] {len(escolhidos)} videos disponiveis em {videos_dir}")
    return escolhidos


# ============================================================
# Etapa 2: Extracao de frames com OpenCV
# ============================================================
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


def extrair_frames(video_path, frames_root, segundos_por_frame, nome_saida, imgsz=640):
    """Extrai 1 frame a cada N segundos via cv2 e redimensiona para imgsz."""
    saida_dir = frames_root / nome_saida
    saida_dir.mkdir(parents=True, exist_ok=True)

    if any(saida_dir.glob("frame_*.jpg")):
        n = len(list(saida_dir.glob("frame_*.jpg")))
        print(f"  [SKIP] {nome_saida}: {n} frames ja extraidos")
        return saida_dir

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERRO] cv2 nao abriu {video_path.name}")
        return saida_dir

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total_frames <= 0:
        print(f"  [ERRO] {nome_saida}: fps={fps}, total_frames={total_frames}")
        cap.release()
        return saida_dir

    intervalo = max(1, int(round(fps * segundos_por_frame)))
    idx_alvo = 0
    salvos = 0

    while idx_alvo < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_alvo)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = redimensionar(frame, imgsz)
        salvos += 1
        out_path = saida_dir / f"frame_{salvos:04d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        idx_alvo += intervalo

    cap.release()
    print(f"  [OK]   {nome_saida}: {salvos} frames extraidos (fps={fps:.1f}, total={total_frames})")
    return saida_dir


# ============================================================
# Etapa 3: Pre-anotacao com YOLO
# ============================================================
def anotar_video(model, frames_dir, labels_root, conf):
    """Roda inferencia em todos os frames do video e salva .txt no formato YOLO.

    Retorna lista de dicts com (frame_path, label_path, deteccoes, classes, conf_media).
    """
    nome = frames_dir.name
    saida_labels = labels_root / nome
    saida_labels.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return []

    resultados = model.predict(
        source=[str(f) for f in frames],
        conf=conf,
        verbose=False,
        stream=True,
    )

    registros = []
    for frame_path, res in zip(frames, resultados):
        label_path = saida_labels / (frame_path.stem + ".txt")

        boxes = res.boxes
        linhas = []
        classes_frame = []
        confs_frame = []

        if boxes is not None and len(boxes) > 0:
            # xywhn = [cx, cy, w, h] normalizados em [0,1]
            xywhn = boxes.xywhn.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for c, (cx, cy, w, h), cf in zip(cls, xywhn, confs):
                linhas.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                classes_frame.append(int(c))
                confs_frame.append(float(cf))

        # Sempre escreve o .txt (vazio se sem deteccoes) para o Roboflow
        label_path.write_text("\n".join(linhas), encoding="utf-8")

        registros.append({
            "video": nome,
            "frame": frame_path.name,
            "frame_path": frame_path,
            "label_path": label_path,
            "n_deteccoes": len(classes_frame),
            "classes": classes_frame,
            "conf_media": (sum(confs_frame) / len(confs_frame)) if confs_frame else 0.0,
        })

    return registros


# ============================================================
# Etapa 4: Organizacao para Roboflow
# ============================================================
def montar_roboflow(registros, roboflow_dir, nomes_por_id):
    """Copia todos os .jpg e .txt para roboflow_import/{images,labels} com nomes unicos."""
    images_dir = roboflow_dir / "images"
    labels_dir = roboflow_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for r in registros:
        novo_nome = f"{r['video']}__{r['frame_path'].stem}"
        dest_img = images_dir / (novo_nome + r["frame_path"].suffix)
        dest_lbl = labels_dir / (novo_nome + ".txt")
        if not dest_img.exists():
            shutil.copy2(r["frame_path"], dest_img)
        if not dest_lbl.exists():
            shutil.copy2(r["label_path"], dest_lbl)
        r["nome_unificado"] = novo_nome

    # classes.txt na ordem dos indices que o modelo aprendeu
    classes_txt = roboflow_dir / "classes.txt"
    classes_txt.write_text(
        "\n".join(nomes_por_id[i] for i in sorted(nomes_por_id)) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] {len(registros)} arquivos copiados para {roboflow_dir}")
    print(f"[OK] classes.txt -> {classes_txt}")


def escrever_csv(registros, csv_path, nomes_por_id):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "nome_frame", "video", "quantidade_deteccoes",
            "classes_detectadas", "confianca_media",
        ])
        for r in registros:
            classes_nomes = ",".join(nomes_por_id.get(c, str(c)) for c in r["classes"])
            w.writerow([
                r.get("nome_unificado", r["frame"]),
                r["video"],
                r["n_deteccoes"],
                classes_nomes,
                f"{r['conf_media']:.4f}",
            ])
    print(f"[OK] Relatorio CSV -> {csv_path}")


def escrever_prioritarios(registros, prioritarios_path, nomes_por_id):
    id_prioritario = next(
        (cid for cid, nome in nomes_por_id.items() if nome == CLASSE_PRIORITARIA_NOME),
        None,
    )
    if id_prioritario is None:
        print(f"[AVISO] classe prioritaria '{CLASSE_PRIORITARIA_NOME}' nao encontrada no modelo")
        prioritarios_path.write_text("", encoding="utf-8")
        return []
    prioritarios = [
        r for r in registros
        if id_prioritario in r["classes"]
    ]
    with prioritarios_path.open("w", encoding="utf-8") as f:
        f.write(f"# Frames com deteccao de '{CLASSE_PRIORITARIA_NOME}' "
                f"(classe {id_prioritario}) - revisar primeiro no Roboflow\n")
        f.write(f"# Total: {len(prioritarios)} frames\n\n")
        for r in prioritarios:
            f.write(r.get("nome_unificado", r["frame"]) + "\n")
    print(f"[OK] Prioritarios -> {prioritarios_path} ({len(prioritarios)} frames)")
    return prioritarios


# ============================================================
# Etapa 5: Relatorio final
# ============================================================
def relatorio_final(registros, prioritarios, n_videos, nomes_por_id):
    total_frames = len(registros)
    com_deteccao = sum(1 for r in registros if r["n_deteccoes"] > 0)
    sem_deteccao = total_frames - com_deteccao

    contagem_classes = Counter()
    for r in registros:
        contagem_classes.update(r["classes"])

    id_prioritario = next(
        (cid for cid, nome in nomes_por_id.items() if nome == CLASSE_PRIORITARIA_NOME),
        None,
    )

    print("\n" + "=" * 60)
    print("RELATORIO FINAL")
    print("=" * 60)
    print(f"Videos processados:         {n_videos}")
    print(f"Frames extraidos:           {total_frames}")
    print(f"Frames com deteccao:        {com_deteccao}")
    print(f"Frames sem deteccao:        {sem_deteccao}")
    print()
    print("Distribuicao de deteccoes por classe:")
    for cid in sorted(nomes_por_id):
        nome = nomes_por_id[cid]
        n = contagem_classes.get(cid, 0)
        marcador = " <- PRIORIDADE" if cid == id_prioritario else ""
        print(f"  [{cid}] {nome:<14} {n:>6}{marcador}")
    print()
    print(f"Frames PRIORITARIOS ({CLASSE_PRIORITARIA_NOME}): {len(prioritarios)}")
    print("=" * 60)


# ============================================================
# Pipeline principal
# ============================================================
def processar_video(video_path, nome_logico_video, frames_root, labels_root, model, segundos, conf):
    """Roda extracao + anotacao para um video. Retorna lista de registros."""
    try:
        frames_dir = extrair_frames(video_path, frames_root, segundos, nome_logico_video)
        return anotar_video(model, frames_dir, labels_root, conf)
    except Exception as e:
        print(f"  [ERRO] Falha processando {video_path.name}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de pre-anotacao YOLO a partir de playlist do YouTube",
    )
    parser.add_argument("--playlist", required=True, help="URL da playlist do YouTube")
    parser.add_argument("--model", required=True, help="Caminho para o best.pt treinado")
    parser.add_argument("--output", default="./output", help="Pasta de saida (default: ./output)")
    parser.add_argument("--fps", type=int, default=3,
                        help="Intervalo em segundos entre frames extraidos (default: 3)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confianca minima da deteccao (default: 0.25)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Pular Etapa 1 e usar videos ja presentes em output/videos")
    args = parser.parse_args()

    # Validacoes (cv2 ja foi importado no topo)
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[ERRO] Modelo nao encontrado: {model_path}")
        sys.exit(1)

    output = Path(args.output).resolve()
    videos_dir = output / "videos"
    frames_root = output / "frames"
    labels_root = output / "labels"
    roboflow_dir = output / "roboflow_import"
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AUTO-ANOTACAO YOLO (PLAYLIST -> ROBOFLOW)")
    print("=" * 60)
    print(f"Playlist: {args.playlist}")
    print(f"Modelo:   {model_path}")
    print(f"Saida:    {output}")
    print(f"FPS:      1 frame / {args.fps}s")
    print(f"Conf:     {args.conf}")
    print("=" * 60)

    # Etapa 1
    if args.skip_download:
        videos_dir.mkdir(parents=True, exist_ok=True)
        videos = selecionar_videos(videos_dir)
        print(f"[INFO] --skip-download ativo. {len(videos)} videos em {videos_dir}")
    else:
        videos = baixar_playlist(args.playlist, videos_dir)

    if not videos:
        print("[ERRO] Nenhum video disponivel para processar.")
        sys.exit(1)

    # Carrega modelo (depois do download para falhar cedo se algo der errado antes)
    print(f"\n[INFO] Carregando modelo: {model_path}")
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    # model.names = {id: nome} aprendido no treino. Source da verdade.
    nomes_por_id = {int(k): v for k, v in model.names.items()}
    print(f"[OK] Modelo carregado. Classes (do modelo):")
    for cid in sorted(nomes_por_id):
        print(f"     [{cid}] {nomes_por_id[cid]}")
    if nomes_por_id != CLASSES:
        print(f"[AVISO] Ordem das classes do modelo difere da constante CLASSES no script.")
        print(f"        Usando a ordem do modelo (correta).")

    # Etapas 2 + 3: por video
    print(f"\n[INFO] Extraindo frames e rodando inferencia em {len(videos)} videos...")
    try:
        from tqdm import tqdm
        iterador = tqdm(videos, desc="videos", unit="video")
    except ImportError:
        iterador = videos
        print("[AVISO] tqdm nao instalado, sem barra de progresso")

    todos_registros = []
    for item in iterador:
        logico, vp = item
        registros = processar_video(
            vp, logico, frames_root, labels_root, model, args.fps, args.conf,
        )
        todos_registros.extend(registros)

    # Etapa 4
    print(f"\n[INFO] Organizando layout do Roboflow em {roboflow_dir}...")
    montar_roboflow(todos_registros, roboflow_dir, nomes_por_id)
    escrever_csv(todos_registros, roboflow_dir / "relatorio.csv", nomes_por_id)
    prioritarios = escrever_prioritarios(
        todos_registros, roboflow_dir / "prioritarios_revisar.txt", nomes_por_id,
    )

    # Etapa 5
    relatorio_final(todos_registros, prioritarios, n_videos=len(videos), nomes_por_id=nomes_por_id)


if __name__ == "__main__":
    main()
