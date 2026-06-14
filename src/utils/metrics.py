"""
Métricas de avaliação para regressão.
9 métricas: MSE, RMSE, MAE, R², MAPE, MedAE, MaxError, EVS, RMSLE.
"""
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score,
    mean_squared_log_error,
)


def compute_all_metrics(y_true, y_pred):
    """
    Calcula todas as 9 métricas de regressão.

    Parameters
    ----------
    y_true : array-like — valores reais
    y_pred : array-like — valores previstos

    Returns
    -------
    dict — nome_metrica → valor
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    maxerr = max_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    # MAPE — evita divisão por zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    # RMSLE — requer valores ≥ 0; se houver negativos, shifta
    y_true_shifted = y_true.copy()
    y_pred_shifted = y_pred.copy()
    min_val = min(y_true_shifted.min(), y_pred_shifted.min())
    if min_val < 0:
        shift = abs(min_val) + 1
        y_true_shifted += shift
        y_pred_shifted += shift
    # Clipa para evitar log(0)
    y_pred_shifted = np.clip(y_pred_shifted, 0, None)
    y_true_shifted = np.clip(y_true_shifted, 0, None)
    try:
        rmsle = np.sqrt(mean_squared_log_error(y_true_shifted, y_pred_shifted))
    except ValueError:
        rmsle = np.nan

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "MedAE": medae,
        "MaxError": maxerr,
        "EVS": evs,
        "RMSLE": rmsle,
    }


def aggregate_metrics(metrics_list):
    """
    Agrega métricas de múltiplas runs (21 execuções).

    Parameters
    ----------
    metrics_list : list[dict] — lista de dicts retornados por compute_all_metrics

    Returns
    -------
    dict — nome_metrica → {"mean", "std", "median", "min", "max"}
    """
    if not metrics_list:
        return {}

    metric_keys = ["MSE", "RMSE", "MAE", "R2", "MAPE", "MedAE", "MaxError", "EVS", "RMSLE"]
    result = {}
    for key in metric_keys:
        if key not in metrics_list[0]:
            continue
        values = np.array([m[key] for m in metrics_list if not np.isnan(m[key])])
        if len(values) == 0:
            result[key] = {"mean": np.nan, "std": np.nan, "median": np.nan,
                           "min": np.nan, "max": np.nan}
        else:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return result
