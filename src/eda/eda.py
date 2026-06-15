"""
Análise Exploratória de Dados (EDA) para os 4 datasets.
Gera estatísticas descritivas, correlações e distribuições.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def run_eda(dataset_index):
    """
    Executa análise exploratória para um dataset.

    Parameters
    ----------
    dataset_index : int — índice do dataset (0-3)
    """
    caminho = config.datasets[dataset_index][1]
    target_col = config.TARGET_COLUMNS[dataset_index]
    ds_name = config.DATASET_NAMES[dataset_index]

    # Carregar
    try:
        df = pd.read_csv(caminho)
    except UnicodeDecodeError:
        df = pd.read_csv(caminho, encoding="latin1")

    print(f"\n{'='*60}")
    print(f"EDA — Dataset {dataset_index}: {ds_name}")
    print(f"{'='*60}")

    # Info básica
    print(f"\nShape: {df.shape}")
    print(f"\nColunas ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        print(f"  {col}: {dtype} | missing={missing} | unique={unique}")

    # Estatísticas descritivas
    print(f"\nEstatísticas descritivas (numéricas):")
    print(df.describe().to_string())

    # Target
    if target_col in df.columns:
        # Se o target veio como string (ex: '$2,000'), converter para numérico
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            df[target_col] = df[target_col].astype(str).str.replace(r'[\$,]', '', regex=True)
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        print(f"\nTarget '{target_col}':")
        print(f"  min:    {df[target_col].min():.4f}")
        print(f"  max:    {df[target_col].max():.4f}")
        print(f"  mean:   {df[target_col].mean():.4f}")
        print(f"  median: {df[target_col].median():.4f}")
        print(f"  std:    {df[target_col].std():.4f}")

    # Salvar gráficos
    eda_dir = os.path.join(config.RESULTS_DIR, "eda", f"dataset_{dataset_index}")
    os.makedirs(eda_dir, exist_ok=True)

    # Distribuição do target
    if target_col in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df[target_col].hist(bins=50, ax=ax, edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribuição de {target_col} — {ds_name}")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Frequência")
        plt.savefig(os.path.join(eda_dir, "target_distribution.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # Correlação (apenas numéricas)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True if len(corr) <= 15 else False,
                    fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                    linewidths=0.5)
        ax.set_title(f"Correlação — {ds_name}")
        plt.savefig(os.path.join(eda_dir, "correlation_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Top correlações com target
        if target_col in corr.columns:
            target_corr = corr[target_col].drop(target_col).sort_values(
                key=abs, ascending=False
            )
            print(f"\nTop correlações com '{target_col}':")
            for feat, val in target_corr.head(10).items():
                print(f"  {feat}: {val:.4f}")

    print(f"\nGráficos salvos em: {eda_dir}")


def run_all_eda():
    """Executa EDA para todos os datasets."""
    for i in range(len(config.datasets)):
        try:
            run_eda(i)
        except Exception as e:
            print(f"Erro no EDA do dataset {i}: {e}")


if __name__ == "__main__":
    run_all_eda()
