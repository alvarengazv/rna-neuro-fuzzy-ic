"""
Neuro-Fuzzy 2: FNN-FCM (Fuzzy Neural Network com Fuzzy C-Means).
Implementação custom sklearn-compatible.

Arquitetura:
  - Geração de regras via Fuzzy C-Means clustering
  - Cada cluster define uma regra fuzzy (centros como protótipos)
  - Funções de pertinência Gaussianas baseadas nos clusters
  - Consequentes Takagi-Sugeno de 1ª ordem (lineares)
  - Treinamento dos consequentes por gradiente descendente

Diferença do ANFIS:
  - ANFIS: MFs definidas por variável, regras = produto cartesiano
  - FNN-FCM: regras definidas por clustering multidimensional do espaço de entrada
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


class FuzzyClusterNNRegressor(BaseEstimator, RegressorMixin):
    """
    Rede Neural Fuzzy baseada em Fuzzy C-Means.

    Parameters
    ----------
    n_clusters : int — número de clusters/regras
    fuzziness : float — expoente de fuzificação do FCM (m > 1)
    mf_type : str — 'gaussian' ou 'bell'
    learning_rate : float — taxa de aprendizado para consequentes
    n_epochs : int — número de épocas de treinamento
    similarity_threshold : float — limiar para ativação mínima das regras
    random_state : int ou None
    """

    def __init__(self, n_clusters=5, fuzziness=2.0, mf_type="gaussian",
                 learning_rate=0.01, n_epochs=100, similarity_threshold=0.1,
                 random_state=None):
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.mf_type = mf_type
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.similarity_threshold = similarity_threshold
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        rng = np.random.RandomState(self.random_state)

        # Limitar clusters ao número de amostras
        k = min(self.n_clusters, n_samples // 2)
        self.effective_n_clusters_ = k

        # 1. Fuzzy C-Means clustering
        self.centers_, self.U_ = self._fuzzy_cmeans(X, k, rng)

        # 2. Calcular sigmas para MFs Gaussianas (baseados nos clusters)
        self.sigmas_ = self._compute_sigmas(X)

        # 3. Inicializar consequentes (Takagi-Sugeno 1ª ordem)
        # Para cada regra i: f_i(x) = θ_i · [x, 1]
        self.consequents_ = rng.randn(k, n_features + 1) * 0.01

        # 4. Treinamento por gradiente descendente
        self._train_consequents(X, y)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Calcular ativação das regras
        activations = self._compute_rule_activation(X)
        # Normalizar
        act_sum = activations.sum(axis=1, keepdims=True) + 1e-10
        act_norm = activations / act_sum
        # Calcular saída de cada regra
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        rule_outputs = X_aug @ self.consequents_.T  # (n_samples, k)
        # Saída final: soma ponderada
        return np.sum(act_norm * rule_outputs, axis=1)

    # ================================================================
    # Fuzzy C-Means
    # ================================================================
    def _fuzzy_cmeans(self, X, k, rng, max_iter=100, tol=1e-5):
        """
        Implementação de Fuzzy C-Means.

        Returns
        -------
        centers : ndarray, shape (k, n_features)
        U : ndarray, shape (n_samples, k) — graus de pertinência
        """
        n_samples = X.shape[0]
        m = self.fuzziness

        # Inicializar graus de pertinência aleatórios
        U = rng.rand(n_samples, k)
        U = U / U.sum(axis=1, keepdims=True)  # normalizar para soma = 1

        centers = np.zeros((k, X.shape[1]))

        for iteration in range(max_iter):
            # Atualizar centros
            Um = U ** m
            for i in range(k):
                num = (Um[:, i:i+1] * X).sum(axis=0)
                den = Um[:, i].sum() + 1e-10
                centers[i] = num / den

            # Atualizar graus de pertinência
            dists = cdist(X, centers, metric="euclidean")
            dists = np.maximum(dists, 1e-10)

            U_new = np.zeros_like(U)
            for i in range(k):
                for j in range(k):
                    U_new[:, i] += (dists[:, i] / dists[:, j]) ** (2 / (m - 1))
            U_new = 1.0 / (U_new + 1e-10)
            U_new = U_new / (U_new.sum(axis=1, keepdims=True) + 1e-10)

            # Verificar convergência
            if np.max(np.abs(U_new - U)) < tol:
                U = U_new
                break
            U = U_new

        return centers, U

    # ================================================================
    # Calcular sigmas baseados nos clusters
    # ================================================================
    def _compute_sigmas(self, X):
        """Calcula sigma por cluster baseado na dispersão dos dados."""
        k = self.effective_n_clusters_
        sigmas = np.zeros(k)

        for i in range(k):
            dists = np.linalg.norm(X - self.centers_[i], axis=1)
            sigmas[i] = max(np.mean(dists), 1e-6)

        # Ajustar pelo threshold de similaridade
        sigmas = sigmas * (1 + self.similarity_threshold)
        return sigmas

    # ================================================================
    # Ativação das regras
    # ================================================================
    def _compute_rule_activation(self, X):
        """
        Calcula a ativação de cada regra para cada amostra.

        Returns
        -------
        activations : ndarray, shape (n_samples, k)
        """
        k = self.effective_n_clusters_
        dists = cdist(X, self.centers_, metric="euclidean")

        if self.mf_type == "gaussian":
            activations = np.exp(-dists ** 2 / (2 * self.sigmas_ ** 2 + 1e-10))
        elif self.mf_type == "bell":
            activations = 1.0 / (1.0 + (dists / (self.sigmas_ + 1e-10)) ** 4 + 1e-10)
        else:
            activations = np.exp(-dists ** 2 / (2 * self.sigmas_ ** 2 + 1e-10))

        return activations

    # ================================================================
    # Treinamento dos consequentes
    # ================================================================
    def _train_consequents(self, X, y):
        """Treina consequentes Takagi-Sugeno por gradiente descendente."""
        n_samples = X.shape[0]
        X_aug = np.hstack([X, np.ones((n_samples, 1))])

        for epoch in range(self.n_epochs):
            # Forward
            activations = self._compute_rule_activation(X)
            act_sum = activations.sum(axis=1, keepdims=True) + 1e-10
            act_norm = activations / act_sum

            rule_outputs = X_aug @ self.consequents_.T  # (n_samples, k)
            y_pred = np.sum(act_norm * rule_outputs, axis=1)

            # Erro
            error = y_pred - y  # (n_samples,)

            # Gradiente dos consequentes
            # dE/dθ_i = Σ_n error_n * w̄_i_n * x_aug_n
            for i in range(self.effective_n_clusters_):
                grad = (error * act_norm[:, i])[:, np.newaxis] * X_aug
                grad_mean = grad.mean(axis=0)
                self.consequents_[i] -= self.learning_rate * grad_mean


def get_model_name():
    return "FNN-FCM"


def get_optuna_search_space(trial):
    """Espaço de busca Optuna para FNN-FCM."""
    n_clusters = trial.suggest_int("n_clusters", 2, 10)
    fuzziness = trial.suggest_float("fuzziness", 1.5, 3.0)
    mf_type = trial.suggest_categorical("mf_type", ["gaussian", "bell"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_epochs = trial.suggest_int("n_epochs", 50, 300, step=50)
    similarity_threshold = trial.suggest_float("similarity_threshold", 0.01, 0.5)

    return {
        "n_clusters": n_clusters,
        "fuzziness": fuzziness,
        "mf_type": mf_type,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "similarity_threshold": similarity_threshold,
    }


def create_model(params, random_state=None):
    """Cria pipeline StandardScaler + FuzzyClusterNNRegressor."""
    fnn = FuzzyClusterNNRegressor(
        n_clusters=params["n_clusters"],
        fuzziness=params["fuzziness"],
        mf_type=params["mf_type"],
        learning_rate=params["learning_rate"],
        n_epochs=params["n_epochs"],
        similarity_threshold=params["similarity_threshold"],
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", fnn),
    ])

    return pipeline
