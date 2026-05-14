"""
Download do dataset do Roboflow no formato YOLOv11.
Estrutura esperada: datasets/deteccao_afogamento/{train,valid,test}/
"""
import os
import sys

# Diretorio raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "deteccao_afogamento")


def baixar_dataset(api_key, workspace, project, version=1):
    from roboflow import Roboflow

    print(f"Conectando ao Roboflow (workspace={workspace}, project={project}, v{version})...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov11", location=DATASET_DIR)
    print(f"[OK] Dataset baixado em: {DATASET_DIR}")
    return dataset


def validar_dataset():
    """Valida que a estrutura do dataset esta correta."""
    erros = []
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(DATASET_DIR, split)
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")

        if not os.path.isdir(images_dir):
            erros.append(f"Diretorio nao encontrado: {images_dir}")
        else:
            n_imgs = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"[OK] {split}/images: {n_imgs} imagens")

        if not os.path.isdir(labels_dir):
            erros.append(f"Diretorio nao encontrado: {labels_dir}")
        else:
            n_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            print(f"[OK] {split}/labels: {n_labels} anotacoes")

    # Verificar data.yaml
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    if not os.path.isfile(yaml_path):
        erros.append(f"data.yaml nao encontrado em {DATASET_DIR}")
    else:
        print(f"[OK] data.yaml encontrado")

    if erros:
        print("\n[ERRO] Problemas encontrados:")
        for e in erros:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n[OK] Dataset validado com sucesso!")


def contar_classes():
    """Conta instancias por classe para verificar desbalanceamento."""
    import yaml

    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    classes = data.get("names", {})
    if isinstance(classes, list):
        classes = {i: name for i, name in enumerate(classes)}

    contagem = {v: 0 for v in classes.values()}

    for split in ["train", "valid", "test"]:
        labels_dir = os.path.join(DATASET_DIR, split, "labels")
        if not os.path.isdir(labels_dir):
            continue
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
            with open(os.path.join(labels_dir, label_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_name = classes.get(class_id, f"classe_{class_id}")
                        contagem[class_name] = contagem.get(class_name, 0) + 1

    print("\nDistribuicao de classes (todas as splits):")
    print("-" * 40)
    total = sum(contagem.values())
    for nome, count in sorted(contagem.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {nome:20s}: {count:5d} ({pct:.1f}%)")
    print(f"  {'TOTAL':20s}: {total:5d}")

    return contagem


def main():
    print("=" * 50)
    print("DOWNLOAD E VALIDACAO DO DATASET")
    print("=" * 50)

    # Verificar se ja existe
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    if os.path.isfile(yaml_path):
        print("[INFO] Dataset ja existe. Pulando download.")
    else:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            api_key = input("Insira sua API Key do Roboflow: ").strip()

        workspace = os.environ.get("ROBOFLOW_WORKSPACE", "pedros-workspace-jewao")
        project = os.environ.get("ROBOFLOW_PROJECT", "deteccao_afogamento")
        version = int(os.environ.get("ROBOFLOW_VERSION", "5"))

        baixar_dataset(api_key, workspace, project, version)

    validar_dataset()
    contar_classes()


if __name__ == "__main__":
    main()
