"""
RNA 1: MLP Regressor (scikit-learn MLPRegressor).
Wrapper com Pipeline (StandardScaler + MLP) e espaço de busca Optuna.
"""
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model_name():
    return "MLP"


def get_optuna_search_space(trial):
    """
    Define o espaço de busca de hiperparâmetros para o MLP via Optuna.

    Returns
    -------
    dict — hiperparâmetros sugeridos
    """
    # Número de camadas ocultas
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # Neurônios por camada
    layers = []
    for i in range(n_layers):
        n_neurons = trial.suggest_int(f"n_neurons_layer_{i}", 16, 256, log=True)
        layers.append(n_neurons)
    hidden_layer_sizes = tuple(layers)

    # Função de ativação
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])

    # Otimizador (se o dataset for grande, lbfgs fica extremamente lento/trava por tentar batch completo)
    dataset_size = 0
    try:
        dataset_size = trial.study.user_attrs.get("dataset_size", 0)
    except Exception:
        pass

    if dataset_size > 20000:
        solver = trial.suggest_categorical("solver", ["adam", "sgd"])
        # Para bases grandes, usar lotes maiores para paralelização eficiente no processador
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        # Limitar épocas para bases grandes (converge rápido com lotes maiores + early stopping)
        max_iter = trial.suggest_int("max_iter", 100, 300, step=50)
    else:
        solver = trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"])
        # Épocas normais
        max_iter = trial.suggest_int("max_iter", 200, 1000, step=100)
        # Batch size (apenas para adam e sgd)
        if solver in ["adam", "sgd"]:
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        else:
            batch_size = "auto"

    # Taxa de aprendizado (apenas para adam e sgd)
    if solver in ["adam", "sgd"]:
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
    else:
        learning_rate_init = 0.001  # default, não usado por lbfgs

    # Regularização L2
    use_regularization = trial.suggest_categorical("use_regularization", [True, False])
    if use_regularization:
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    else:
        alpha = 0.0

    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "batch_size": batch_size,
        "alpha": alpha,
    }


def create_model(params, random_state=None):
    """
    Cria o pipeline StandardScaler + MLPRegressor com os parâmetros dados.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        solver=params["solver"],
        learning_rate_init=params["learning_rate_init"],
        max_iter=params["max_iter"],
        batch_size=params["batch_size"],
        alpha=params["alpha"],
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", mlp),
    ])

    return pipeline
