"""
Orquestração dos experimentos: otimização com Optuna + 21 runs independentes.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, KFold, train_test_split

# Adiciona src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.metrics import compute_all_metrics, aggregate_metrics
from utils import plots
from preprocessing.preprocessing import load_and_preprocess

# Importar módulos dos modelos
from models.rna import mlp, rbf
from models.neuro_fuzzy import anfis, fnn_fcm

# Silenciar warnings desnecessários durante otimização
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Lista de módulos de modelo (mesma ordem de MODEL_NAMES)
MODEL_MODULES = [mlp, rbf, anfis, fnn_fcm]


def _print_progress(current, total, prefix="", suffix="", bar_length=30):
    """Imprime barra de progresso in-place no terminal."""
    pct = current / total if total > 0 else 1
    filled = int(bar_length * pct)
    bar = "#" * filled + "-" * (bar_length - filled)
    line = f"\r    {prefix} [{bar}] {current}/{total} ({pct*100:.0f}%) {suffix}"
    print(line, end="", flush=True)
    if current >= total:
        print()  # nova linha ao finalizar


def optimize_hyperparams(model_module, X_train, y_train, n_trials, n_folds, random_state=42):
    """
    Otimiza hiperparâmetros usando Optuna com validação cruzada.

    Parameters
    ----------
    model_module : módulo do modelo (rna1, rna2, neuro_fuzzy1, neuro_fuzzy2)
    X_train : np.ndarray
    y_train : np.ndarray
    n_trials : int
    n_folds : int
    random_state : int

    Returns
    -------
    best_params : dict
    study : optuna.Study
    """
    model_name = model_module.get_model_name()
    n_samples = X_train.shape[0]

    # --- Otimização 1: subamostragem para datasets grandes ---
    _MAX_OPTUNA_SAMPLES = 50000
    if n_samples > _MAX_OPTUNA_SAMPLES:
        rng_sub = np.random.RandomState(random_state)
        sub_idx = rng_sub.choice(n_samples, _MAX_OPTUNA_SAMPLES, replace=False)
        X_optuna = X_train[sub_idx]
        y_optuna = y_train[sub_idx]
    else:
        X_optuna = X_train
        y_optuna = y_train

    # --- Otimização 2: reduzir folds para datasets grandes ---
    if n_samples > 100000 and n_folds > 3:
        n_folds = 3

    def objective(trial):
        params = model_module.get_optuna_search_space(trial, dataset_size=n_samples)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        scores = []

        for train_idx, val_idx in kf.split(X_optuna):
            X_tr, X_val = X_optuna[train_idx], X_optuna[val_idx]
            y_tr, y_val = y_optuna[train_idx], y_optuna[val_idx]

            try:
                model = model_module.create_model(params, random_state=random_state)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)

                # Usar RMSE como métrica de otimização (minimizar)
                rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                scores.append(rmse)
            except Exception as e:
                # Se o modelo falhar, retornar valor alto
                return float("inf")

        return np.mean(scores)

    # Criar estudo Optuna
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.set_user_attr("dataset_size", int(X_train.shape[0]))

    # Log das otimizações aplicadas
    opt_info = []
    if n_samples > _MAX_OPTUNA_SAMPLES:
        opt_info.append(f"sub={_MAX_OPTUNA_SAMPLES}")
    if n_samples > 100000:
        opt_info.append(f"{n_folds}-fold")
    opt_str = f" [{', '.join(opt_info)}]" if opt_info else ""

    print(f"\n  → Otimizando {model_name} ({n_trials} trials, {n_folds}-fold CV){opt_str}...")

    # Callback de progresso para Optuna
    def _optuna_callback(study, trial):
        n_complete = len(study.trials)
        best = study.best_value
        best_str = f"| Best RMSE: {best:.4f}" if best < float("inf") else ""
        _print_progress(n_complete, n_trials, prefix="Trials", suffix=best_str)

    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_optuna_callback])
    elapsed = time.time() - start

    best_params = study.best_params
    # Reconstruir params completos (para lidar com params condicionais como hidden_layer_sizes)
    best_trial = study.best_trial
    best_params_full = model_module.get_optuna_search_space(best_trial, dataset_size=n_samples)

    print(f"    Melhor RMSE (CV): {study.best_value:.4f}")
    print(f"    Tempo: {elapsed:.1f}s")
    print(f"    Params: {best_params_full}")

    return best_params_full, study


def run_single_experiment(model_module, X_train, y_train, X_test, y_test,
                          params, seed):
    """
    Executa uma run do modelo com parâmetros fixos e seed específica.

    Returns
    -------
    metrics : dict — 9 métricas calculadas no conjunto de teste
    y_pred : np.ndarray — predições no teste
    """
    model = model_module.create_model(params, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)
    return metrics, y_pred


def run_all_experiments(dataset_indices=None, model_indices=None,
                        n_trials=None, n_runs=None, n_folds=None):
    """
    Executa o pipeline completo de experimentos.

    Parameters
    ----------
    dataset_indices : list[int] ou None — quais datasets rodar (0-3)
    model_indices : list[int] ou None — quais modelos rodar (0-3)
    n_trials : int ou None — override do número de trials
    n_runs : int ou None — override do número de runs
    n_folds : int ou None — override do número de folds
    """
    if dataset_indices is None:
        dataset_indices = list(range(len(config.datasets)))
    if model_indices is None:
        model_indices = list(range(len(MODEL_MODULES)))
    if n_trials is None:
        n_trials = config.N_TRIALS
    if n_runs is None:
        n_runs = config.N_RUNS
    if n_folds is None:
        n_folds = config.N_FOLDS

    # Diretório de resultados
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Armazenar todos os resultados
    all_results = {}   # {ds_idx: DataFrame}
    all_summaries = {} # {ds_idx: dict}

    for ds_idx in dataset_indices:
        ds_name = config.DATASET_NAMES[ds_idx]
        print(f"\n{'#'*70}")
        print(f"# DATASET {ds_idx}: {ds_name}")
        print(f"{'#'*70}")

        # Carregar e preprocessar
        try:
            X, y, feature_names = load_and_preprocess(ds_idx)
        except Exception as e:
            print(f"  ERRO ao carregar dataset {ds_idx}: {e}")
            continue

        # Split treino/teste fixo (para comparação justa entre modelos)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )

        print(f"  Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

        ds_results = []   # Lista de dicts por run
        ds_dir = os.path.join(config.RESULTS_DIR, f"dataset_{ds_idx}")
        os.makedirs(ds_dir, exist_ok=True)

        for mod_idx in model_indices:
            model_module = MODEL_MODULES[mod_idx]
            model_name = model_module.get_model_name()

            print(f"\n  {'='*50}")
            print(f"  MODELO: {model_name}")
            print(f"  {'='*50}")

            # 1. Otimizar hiperparâmetros com Optuna
            try:
                best_params, study = optimize_hyperparams(
                    model_module, X_train, y_train,
                    n_trials=n_trials, n_folds=n_folds
                )
            except Exception as e:
                print(f"  ERRO na otimização de {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Salvar curva de convergência
            try:
                plots.plot_convergence(
                    study, f"Convergência Optuna — {model_name} — {ds_name}",
                    os.path.join(ds_dir, f"convergence_{model_name.lower().replace(' ', '_')}.png")
                )
            except Exception:
                pass

            # Salvar best params
            params_path = os.path.join(ds_dir, f"best_params_{model_name.lower().replace(' ', '_')}.txt")
            with open(params_path, "w") as f:
                for k, v in best_params.items():
                    f.write(f"{k}: {v}\n")

            # 2. Executar 21 runs independentes
            print(f"\n  → Executando {n_runs} runs independentes...")
            run_metrics = []
            last_y_pred = None
            run_start = time.time()

            for run in range(n_runs):
                seed = run  # Seeds 0 a 20
                try:
                    metrics, y_pred = run_single_experiment(
                        model_module, X_train, y_train,
                        X_test, y_test, best_params, seed
                    )
                    metrics["model"] = model_name
                    metrics["run"] = run
                    run_metrics.append(metrics)
                    last_y_pred = y_pred

                    elapsed_runs = time.time() - run_start
                    avg_per_run = elapsed_runs / (run + 1)
                    remaining = avg_per_run * (n_runs - run - 1)
                    eta_str = f"| ETA: {remaining:.0f}s" if remaining > 0 else ""
                    _print_progress(
                        run + 1, n_runs, prefix="Runs",
                        suffix=f"| RMSE={metrics['RMSE']:.4f} {eta_str}"
                    )

                except Exception as e:
                    _print_progress(run + 1, n_runs, prefix="Runs", suffix=f"| ERRO")
                    print(f"\n    Run {run}: ERRO — {e}")

            if not run_metrics:
                print(f"  Nenhuma run concluída para {model_name}")
                continue

            ds_results.extend(run_metrics)

            # 3. Agregar métricas
            agg = aggregate_metrics(run_metrics)
            print(f"\n  → Resultados agregados ({model_name}):")
            for metric_name, vals in agg.items():
                if metric_name in ["RMSE", "MAE", "R2", "MAPE"]:
                    print(f"    {metric_name}: {vals['mean']:.4f} ± {vals['std']:.4f}")

            # 4. Gráficos individuais (último run)
            if last_y_pred is not None:
                plots.plot_predictions_vs_actual(
                    y_test, last_y_pred,
                    f"Real vs Previsto — {model_name} — {ds_name}",
                    os.path.join(ds_dir, f"scatter_{model_name.lower().replace(' ', '_')}.png")
                )
                plots.plot_residuals(
                    y_test, last_y_pred,
                    f"Resíduos — {model_name} — {ds_name}",
                    os.path.join(ds_dir, f"residuals_{model_name.lower().replace(' ', '_')}.png")
                )

        # 5. Salvar resultados do dataset
        if ds_results:
            df_results = pd.DataFrame(ds_results)
            all_results[ds_idx] = df_results

            # CSV com todos os resultados
            csv_path = os.path.join(ds_dir, "all_results.csv")
            df_results.to_csv(csv_path, index=False)
            print(f"\n  Resultados salvos em: {csv_path}")

            # Tabela resumo
            summary_path = os.path.join(ds_dir, "summary.csv")
            metric_cols = [c for c in df_results.columns if c not in ["model", "run"]]
            summary = df_results.groupby("model")[metric_cols].agg(["mean", "std"])
            summary.to_csv(summary_path)
            print(f"  Resumo salvo em: {summary_path}")

    # 6. Gráficos comparativos globais
    if all_results:
        print(f"\n{'#'*70}")
        print("# GERANDO GRÁFICOS COMPARATIVOS")
        print(f"{'#'*70}")
        plots.generate_all_plots(all_results, config.RESULTS_DIR)
        print("  Gráficos gerados com sucesso!")

    # 7. Tabela comparativa global
    _generate_global_summary(all_results)

    return all_results


def _generate_global_summary(all_results):
    """Gera tabela comparativa final com todos os datasets e modelos."""
    if not all_results:
        return

    rows = []
    for ds_idx, df in all_results.items():
        ds_name = config.DATASET_NAMES[ds_idx]
        metric_cols = [c for c in df.columns if c not in ["model", "run"]]

        for model_name in df["model"].unique():
            model_df = df[df["model"] == model_name]
            row = {"Dataset": ds_name, "Modelo": model_name}
            for col in metric_cols:
                values = model_df[col].dropna()
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std()
            rows.append(row)

    global_df = pd.DataFrame(rows)
    global_path = os.path.join(config.RESULTS_DIR, "global_summary.csv")
    global_df.to_csv(global_path, index=False)
    print(f"\n  Resumo global salvo em: {global_path}")

    # Imprimir tabela no terminal
    print("\n" + "=" * 100)
    print("RESUMO GLOBAL — Média ± Std (21 runs)")
    print("=" * 100)
    for _, row in global_df.iterrows():
        print(f"\n  {row['Dataset']} | {row['Modelo']}")
        for col in ["RMSE", "MAE", "R2", "MAPE"]:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            if mean_col in row and std_col in row:
                print(f"    {col}: {row[mean_col]:.4f} ± {row[std_col]:.4f}")
