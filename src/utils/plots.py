"""
Funções de visualização para resultados de regressão.
Gráficos comparativos salvos automaticamente em results/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo global
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})


def _ensure_dir(path):
    """Cria o diretório se não existir."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_predictions_vs_actual(y_true, y_pred, title, save_path):
    """
    Scatter plot: valores reais vs previstos com linha ideal.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolors="none")
    lims = [
        min(min(y_true), min(y_pred)),
        max(max(y_true), max(y_pred)),
    ]
    ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Valor Real")
    ax.set_ylabel("Valor Previsto")
    ax.set_title(title)
    ax.legend()
    plt.savefig(save_path)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title, save_path):
    """
    Histograma dos resíduos (erro = real - previsto).
    """
    _ensure_dir(save_path)
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Resíduo (Real - Previsto)")
    ax.set_ylabel("Frequência")
    ax.set_title(title)
    plt.savefig(save_path)
    plt.close(fig)


def plot_boxplot_comparison(results_df, metric, dataset_name, save_path):
    """
    Boxplot das 21 runs para cada modelo num dado dataset.

    Parameters
    ----------
    results_df : pd.DataFrame — colunas: ['model', 'run', metric, ...]
    metric : str — nome da métrica (e.g. 'RMSE')
    dataset_name : str
    save_path : str
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots()
    sns.boxplot(data=results_df, x="model", y=metric, ax=ax, palette="Set2")
    ax.set_title(f"{metric} — {dataset_name} (21 runs)")
    ax.set_xlabel("Modelo")
    ax.set_ylabel(metric)
    plt.xticks(rotation=15)
    plt.savefig(save_path)
    plt.close(fig)


def plot_heatmap_metrics(summary_df, dataset_name, save_path):
    """
    Heatmap: modelos (linhas) × métricas (colunas) com valores médios.

    Parameters
    ----------
    summary_df : pd.DataFrame — index=modelos, columns=métricas, values=médias
    dataset_name : str
    save_path : str
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(summary_df, annot=True, fmt=".4f", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title(f"Métricas Médias — {dataset_name}")
    ax.set_ylabel("Modelo")
    plt.savefig(save_path)
    plt.close(fig)


def plot_convergence(study, title, save_path):
    """
    Curva de convergência do Optuna (melhor valor por trial).
    """
    _ensure_dir(save_path)
    trials = study.trials
    best_values = []
    current_best = float("inf")
    for t in trials:
        if t.value is not None and t.value < current_best:
            current_best = t.value
        best_values.append(current_best)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(best_values) + 1), best_values, marker="o",
            markersize=3, linewidth=1.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Melhor RMSE (CV)")
    ax.set_title(title)
    plt.savefig(save_path)
    plt.close(fig)


def plot_bar_comparison(results_df, metric, dataset_name, save_path):
    """
    Gráfico de barras com error bars (média ± std) por modelo.

    Parameters
    ----------
    results_df : pd.DataFrame — colunas: ['model', metric, ...]
    """
    _ensure_dir(save_path)
    grouped = results_df.groupby("model")[metric]
    means = grouped.mean()
    stds = grouped.std()

    fig, ax = plt.subplots()
    x = range(len(means))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=sns.color_palette("Set2"),
                  edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=15)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — {dataset_name} (média ± std, 21 runs)")

    # Adicionar valores nas barras
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=8)

    plt.savefig(save_path)
    plt.close(fig)


def generate_all_plots(all_results, results_dir):
    """
    Gera todos os gráficos comparativos para todos os datasets.

    Parameters
    ----------
    all_results : dict — {dataset_idx: pd.DataFrame com colunas [model, run, MSE, RMSE, ...]}
    results_dir : str — diretório base para salvar
    """
    metrics_to_plot = ["RMSE", "MAE", "R2", "MAPE"]

    for ds_idx, df in all_results.items():
        from config import DATASET_NAMES
        ds_name = DATASET_NAMES[ds_idx]
        ds_dir = os.path.join(results_dir, f"dataset_{ds_idx}")
        os.makedirs(ds_dir, exist_ok=True)

        for metric in metrics_to_plot:
            if metric in df.columns:
                # Boxplot
                plot_boxplot_comparison(
                    df, metric, ds_name,
                    os.path.join(ds_dir, f"boxplot_{metric.lower()}.png")
                )
                # Barras
                plot_bar_comparison(
                    df, metric, ds_name,
                    os.path.join(ds_dir, f"bar_{metric.lower()}.png")
                )

        # Heatmap de métricas médias
        all_metrics = ["MSE", "RMSE", "MAE", "R2", "MAPE", "MedAE",
                       "MaxError", "EVS", "RMSLE"]
        available = [m for m in all_metrics if m in df.columns]
        if available:
            summary = df.groupby("model")[available].mean()
            plot_heatmap_metrics(
                summary, ds_name,
                os.path.join(ds_dir, "heatmap_metrics.png")
            )
