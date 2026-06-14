"""
Consolidação de resultados de múltiplos computadores.
Lê os CSVs individuais de cada dataset e gera gráficos e resumo globais.
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Adiciona src ao path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import config
from utils import plots
from experiments import _generate_global_summary


def main():
    print("\n" + "=" * 60)
    print("  📊 CONSOLIDADOR DE RESULTADOS MULTI-MÁQUINA")
    print("=" * 60)
    print(f"  Procurando pastas em: {config.RESULTS_DIR}\n")

    all_results = {}

    for i in range(len(config.datasets)):
        ds_name = config.DATASET_NAMES[i]
        csv_path = os.path.join(config.RESULTS_DIR, f"dataset_{i}", "all_results.csv")

        if os.path.exists(csv_path):
            print(f"  [✓] Dataset {i} ({ds_name}): Encontrado!")
            try:
                df = pd.read_csv(csv_path)
                all_results[i] = df
            except Exception as e:
                print(f"      Erro ao carregar CSV: {e}")
        else:
            print(f"  [ ] Dataset {i} ({ds_name}): Não encontrado (caminho: {csv_path})")

    if not all_results:
        print("\n  ❌ ERRO: Nenhum arquivo 'all_results.csv' foi encontrado!")
        print("  Certifique-se de copiar as pastas 'dataset_0', 'dataset_1', etc.")
        print(f"  para dentro de: {config.RESULTS_DIR}")
        return

    print(f"\n  ✓ Carregados resultados de {len(all_results)} datasets.")

    # 1. Gerar gráficos comparativos globais
    print("\n  → Gerando gráficos comparativos globais...")
    try:
        plots.generate_all_plots(all_results, config.RESULTS_DIR)
        print("      Gráficos gerados com sucesso!")
    except Exception as e:
        print(f"      Erro ao gerar gráficos: {e}")

    # 2. Gerar tabela resumo global
    print("  → Gerando tabela comparativa e resumo global...")
    try:
        _generate_global_summary(all_results)
    except Exception as e:
        print(f"      Erro ao gerar resumo global: {e}")

    print("\n" + "=" * 60)
    print("  🎉 CONSOLIDAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
