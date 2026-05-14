"""
Compara metricas entre modelos treinados (yolo11n vs yolo26n vs yolo26s).
Gera tabela comparativa e grafico de barras.
"""
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def carregar_metricas(modelo_nome):
    """Carrega metricas do CSV de resultados do treino."""
    results_csv = os.path.join(RUNS_DIR, modelo_nome, "results.csv")
    if not os.path.isfile(results_csv):
        print(f"[ERRO] Resultados nao encontrados: {results_csv}")
        return None

    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # Ultima epoca (melhor resultado apos early stopping)
    last = rows[-1]

    # Normalizar nomes das colunas (remover espacos)
    last = {k.strip(): v.strip() for k, v in last.items()}

    metricas = {}
    # Mapear nomes de colunas possiveis
    mapa = {
        "Precision": ["metrics/precision(B)", "precision"],
        "Recall": ["metrics/recall(B)", "recall"],
        "mAP@50": ["metrics/mAP50(B)", "mAP50"],
        "mAP@50-95": ["metrics/mAP50-95(B)", "mAP50-95"],
    }

    for nome_metrica, colunas_possiveis in mapa.items():
        for col in colunas_possiveis:
            if col in last:
                metricas[nome_metrica] = float(last[col])
                break

    # Epoca final
    for col in ["epoch", "Epoch"]:
        if col in last:
            metricas["Epochs"] = int(float(last[col])) + 1
            break

    return metricas


def exibir_comparativo(resultados):
    """Exibe tabela comparativa das metricas."""
    print(f"\n{'=' * 70}")
    print("COMPARATIVO DE MODELOS")
    print(f"{'=' * 70}")

    metricas_nomes = ["Precision", "Recall", "mAP@50", "mAP@50-95", "Epochs"]

    # Cabecalho
    header = f"{'Metrica':20s}"
    for modelo in resultados:
        header += f" | {modelo:>12s}"
    if len(resultados) == 2:
        header += f" | {'Diferenca':>12s}"
    print(header)
    print("-" * len(header))

    # Dados
    modelos = list(resultados.keys())
    for metrica in metricas_nomes:
        linha = f"{metrica:20s}"
        valores = []
        for modelo in modelos:
            val = resultados[modelo].get(metrica, None)
            if val is not None:
                if metrica == "Epochs":
                    linha += f" | {int(val):>12d}"
                else:
                    linha += f" | {val:>12.4f}"
                valores.append(val)
            else:
                linha += f" | {'N/A':>12s}"
                valores.append(None)

        # Diferenca entre os dois modelos
        if len(valores) == 2 and all(v is not None for v in valores):
            diff = valores[1] - valores[0]
            sinal = "+" if diff > 0 else ""
            if metrica == "Epochs":
                linha += f" | {sinal}{int(diff):>11d}"
            else:
                linha += f" | {sinal}{diff:>11.4f}"

        print(linha)

    print(f"\n[INFO] Valores positivos na diferenca favorecem {modelos[-1]}.")


def gerar_grafico(resultados):
    """Gera grafico de barras comparativo."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[AVISO] matplotlib nao instalado. Pulando grafico.")
        return

    metricas_plot = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    modelos = list(resultados.keys())

    x = range(len(metricas_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, modelo in enumerate(modelos):
        valores = [resultados[modelo].get(m, 0) for m in metricas_plot]
        offset = (i - len(modelos) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], valores, width, label=modelo)
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Valor')
    ax.set_title('Comparativo: Deteccao de Afogamento - ' + ' vs '.join(modelos))
    ax.set_xticks(list(x))
    ax.set_xticklabels(metricas_plot)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    output_path = os.path.join(RUNS_DIR, "comparativo_modelos.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n[OK] Grafico salvo em: {output_path}")
    plt.close()


def main():
    print("Carregando metricas dos modelos treinados...\n")

    resultados = {}
    for modelo in ["yolo11n", "yolo26n", "yolo26s"]:
        metricas = carregar_metricas(modelo)
        if metricas:
            resultados[modelo] = metricas
            print(f"[OK] {modelo}: metricas carregadas")
        else:
            print(f"[AVISO] {modelo}: sem resultados (nao treinado?)")

    if not resultados:
        print("[ERRO] Nenhum modelo treinado encontrado.")
        sys.exit(1)

    exibir_comparativo(resultados)
    gerar_grafico(resultados)


if __name__ == "__main__":
    main()
