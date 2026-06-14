"""
RNA 2: RBF Network (Radial Basis Function Network).
Implementação custom sklearn-compatible.

Arquitetura:
  - Camada de entrada (features)
  - Camada oculta: neurônios RBF com ativação Gaussiana
    centros definidos via K-Means
  - Camada de saída: combinação linear (Ridge regression)
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


class RBFNetRegressor(BaseEstimator, RegressorMixin):
    """
    Rede RBF para regressão.

    Parameters
    ----------
    n_centers : int — número de centros RBF (neurônios na camada oculta)
    sigma : float — largura das Gaussianas (se sigma_mode='fixed')
    sigma_mode : str — 'fixed' ou 'per_center'
    alpha : float — regularização Ridge para pesos de saída
    random_state : int ou None
    """

    def __init__(self, n_centers=50, sigma=1.0, sigma_mode="fixed",
                 alpha=1e-3, random_state=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.sigma_mode = sigma_mode
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        # Limitar n_centers ao número de amostras
        k = min(self.n_centers, n_samples)

        # 1. Definir centros via K-Means
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        self.centers_ = kmeans.cluster_centers_  # (k, n_features)

        # 2. Calcular sigmas
        if self.sigma_mode == "per_center":
            # Sigma baseado na distância média ao centro mais próximo
            dists = cdist(self.centers_, self.centers_)
            np.fill_diagonal(dists, np.inf)
            self.sigmas_ = np.min(dists, axis=1)  # distância ao vizinho mais próximo
            self.sigmas_ = np.maximum(self.sigmas_, 1e-8)  # evitar zero
        else:
            self.sigmas_ = np.full(k, self.sigma)

        # 3. Construir matriz de ativação Φ
        Phi = self._compute_activation(X)

        # 4. Ridge regression: w = (Φᵀ Φ + α I)⁻¹ Φᵀ y
        PhiT_Phi = Phi.T @ Phi + self.alpha * np.eye(k + 1)
        PhiT_y = Phi.T @ y
        self.weights_ = np.linalg.solve(PhiT_Phi, PhiT_y)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Phi = self._compute_activation(X)
        return Phi @ self.weights_

    def _compute_activation(self, X):
        """
        Calcula a matriz de ativação Gaussiana + bias.

        Returns
        -------
        Phi : np.ndarray, shape (n_samples, n_centers + 1)
        """
        # Distâncias euclidianas: (n_samples, n_centers)
        dists = cdist(X, self.centers_, metric="euclidean")

        # Ativação Gaussiana: exp(-d² / (2σ²))
        Phi = np.exp(-dists ** 2 / (2 * self.sigmas_ ** 2))

        # Adicionar bias (coluna de 1s)
        bias = np.ones((X.shape[0], 1))
        Phi = np.hstack([Phi, bias])

        return Phi


def get_model_name():
    return "RBF Network"


def get_optuna_search_space(trial):
    """
    Espaço de busca Optuna para a RBF Network.
    """
    n_centers = trial.suggest_int("n_centers", 10, 200, log=True)
    sigma_mode = trial.suggest_categorical("sigma_mode", ["fixed", "per_center"])

    if sigma_mode == "fixed":
        sigma = trial.suggest_float("sigma", 0.1, 10.0, log=True)
    else:
        sigma = 1.0  # ignorado quando per_center

    alpha = trial.suggest_float("alpha", 1e-5, 1e1, log=True)

    return {
        "n_centers": n_centers,
        "sigma": sigma,
        "sigma_mode": sigma_mode,
        "alpha": alpha,
    }


def create_model(params, random_state=None):
    """
    Cria pipeline StandardScaler + RBFNetRegressor.
    """
    rbf = RBFNetRegressor(
        n_centers=params["n_centers"],
        sigma=params["sigma"],
        sigma_mode=params["sigma_mode"],
        alpha=params["alpha"],
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", rbf),
    ])

    return pipeline
