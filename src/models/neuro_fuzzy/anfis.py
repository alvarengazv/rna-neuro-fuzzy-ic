"""
Neuro-Fuzzy 1: ANFIS (Adaptive Neuro-Fuzzy Inference System).
Implementação Takagi-Sugeno de 1ª ordem, sklearn-compatible.

Arquitetura (5 camadas):
  L1 — Fuzzificação: funções de pertinência (Gaussian/Bell/Triangular)
  L2 — Força de disparo das regras (produto dos graus de pertinência)
  L3 — Normalização das forças de disparo
  L4 — Consequentes lineares: f_i = a_i·x + b_i
  L5 — Saída: soma ponderada

Aprendizado híbrido:
  Forward: LSE para consequentes (camada 4)
  Backward: Gradiente descendente para premissas (camada 1)
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ANFISRegressor(BaseEstimator, RegressorMixin):
    """
    ANFIS para regressão — Takagi-Sugeno de 1ª ordem.

    Parameters
    ----------
    n_mfs : int — número de funções de pertinência por variável de entrada
    mf_type : str — 'gaussian', 'bell' ou 'triangular'
    learning_rate : float — taxa de aprendizado para premissas
    n_epochs : int — número de épocas de treinamento
    lambda_reg : float — regularização L2 nos consequentes (LSE)
    random_state : int ou None
    """

    def __init__(self, n_mfs=3, mf_type="gaussian", learning_rate=0.01,
                 n_epochs=100, lambda_reg=1e-4, random_state=None):
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.lambda_reg = lambda_reg
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)

        # Limitar n_mfs para features muito altas (evitar explosão combinatória)
        # Se n_features > 10, usar no máximo 2 MFs para manter viável
        effective_n_mfs = self.n_mfs
        if n_features > 10:
            effective_n_mfs = min(self.n_mfs, 2)
        self.effective_n_mfs_ = effective_n_mfs

        # Selecionar features mais relevantes se necessário
        # Para evitar explosão de regras (n_mfs^n_features)
        max_input_features = 6  # Máximo de features para ANFIS
        if n_features > max_input_features:
            # Selecionar por correlação com target
            correlations = np.array([
                abs(np.corrcoef(X[:, i], y)[0, 1])
                for i in range(n_features)
            ])
            correlations = np.nan_to_num(correlations, 0)
            self.selected_features_ = np.argsort(correlations)[-max_input_features:]
            X = X[:, self.selected_features_]
            n_features = X.shape[1]
        else:
            self.selected_features_ = np.arange(n_features)

        self.n_features_ = n_features
        n_rules = effective_n_mfs ** n_features

        # Limitar regras para evitar problemas de memória
        if n_rules > 500:
            effective_n_mfs = max(2, int(500 ** (1.0 / n_features)))
            self.effective_n_mfs_ = effective_n_mfs
            n_rules = effective_n_mfs ** n_features

        self.n_rules_ = n_rules

        # ---- Inicializar parâmetros das MFs (premissas) ----
        self._init_mf_params(X, rng)

        # ---- Treinamento híbrido ----
        _MAX_EPOCH_SAMPLES = 10000  # Subamostragem por época para datasets grandes
        _PROGRESS_INTERVAL = max(1, self.n_epochs // 10)  # Mostrar progresso a cada ~10%
        for epoch in range(self.n_epochs):
            # Subamostragem estocástica por época (varia a cada época)
            if n_samples > _MAX_EPOCH_SAMPLES:
                epoch_idx = rng.choice(n_samples, _MAX_EPOCH_SAMPLES, replace=False)
                X_ep = X[epoch_idx]
                y_ep = y[epoch_idx]
            else:
                X_ep = X
                y_ep = y

            # Forward pass
            mu = self._fuzzify(X_ep)               # (n_batch, n_features, n_mfs)
            w = self._compute_firing(mu)            # (n_batch, n_rules)
            w_norm = self._normalize_firing(w)      # (n_batch, n_rules)

            # LSE para consequentes (forward)
            self._fit_consequents_lse(X_ep, y_ep, w_norm)

            # Predição com consequentes atuais
            y_pred = self._predict_with_params(X_ep, w_norm)
            error = y_ep - y_pred

            # Backward pass — atualizar premissas por gradiente
            self._update_premises(X_ep, y_ep, mu, w, w_norm, error)

            # Progresso de épocas
            if self.n_epochs >= 20 and ((epoch + 1) % _PROGRESS_INTERVAL == 0 or epoch == 0):
                rmse = np.sqrt(np.mean(error ** 2))
                pct = (epoch + 1) / self.n_epochs * 100
                filled = int(20 * (epoch + 1) / self.n_epochs)
                bar = "#" * filled + "-" * (20 - filled)
                print(f"\r        Epochs [{bar}] {epoch+1}/{self.n_epochs} ({pct:.0f}%) | RMSE: {rmse:.4f}", end="", flush=True)

        if self.n_epochs >= 20:
            print()  # nova linha ao finalizar épocas

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        # Selecionar features
        X = X[:, self.selected_features_]

        mu = self._fuzzify(X)
        w = self._compute_firing(mu)
        w_norm = self._normalize_firing(w)
        return self._predict_with_params(X, w_norm)

    # ================================================================
    # Inicialização das funções de pertinência
    # ================================================================
    def _init_mf_params(self, X, rng):
        """Inicializa centros e larguras das MFs por feature."""
        n_mfs = self.effective_n_mfs_

        if self.mf_type == "gaussian":
            # Centros: distribuídos uniformemente no range de cada feature
            # Sigmas: baseados no espaçamento entre centros
            self.centers_ = np.zeros((self.n_features_, n_mfs))
            self.sigmas_ = np.zeros((self.n_features_, n_mfs))
            for j in range(self.n_features_):
                x_min, x_max = X[:, j].min(), X[:, j].max()
                self.centers_[j] = np.linspace(x_min, x_max, n_mfs)
                spacing = (x_max - x_min) / max(n_mfs - 1, 1)
                self.sigmas_[j] = np.full(n_mfs, max(spacing / 2, 1e-6))

        elif self.mf_type == "bell":
            self.centers_ = np.zeros((self.n_features_, n_mfs))
            self.widths_ = np.zeros((self.n_features_, n_mfs))
            self.slopes_ = np.full((self.n_features_, n_mfs), 2.0)
            for j in range(self.n_features_):
                x_min, x_max = X[:, j].min(), X[:, j].max()
                self.centers_[j] = np.linspace(x_min, x_max, n_mfs)
                spacing = (x_max - x_min) / max(n_mfs - 1, 1)
                self.widths_[j] = np.full(n_mfs, max(spacing / 2, 1e-6))

        elif self.mf_type == "triangular":
            self.lefts_ = np.zeros((self.n_features_, n_mfs))
            self.centers_ = np.zeros((self.n_features_, n_mfs))
            self.rights_ = np.zeros((self.n_features_, n_mfs))
            for j in range(self.n_features_):
                x_min, x_max = X[:, j].min(), X[:, j].max()
                c = np.linspace(x_min, x_max, n_mfs)
                self.centers_[j] = c
                spacing = (x_max - x_min) / max(n_mfs - 1, 1)
                self.lefts_[j] = c - spacing
                self.rights_[j] = c + spacing

    # ================================================================
    # L1 — Fuzzificação
    # ================================================================
    def _fuzzify(self, X):
        """
        Calcula graus de pertinência.

        Returns
        -------
        mu : ndarray, shape (n_samples, n_features, n_mfs)
        """
        n_samples = X.shape[0]
        n_mfs = self.effective_n_mfs_

        mu = np.zeros((n_samples, self.n_features_, n_mfs))

        if self.mf_type == "gaussian":
            for j in range(self.n_features_):
                for k in range(n_mfs):
                    diff = X[:, j] - self.centers_[j, k]
                    mu[:, j, k] = np.exp(-(diff ** 2) / (2 * self.sigmas_[j, k] ** 2 + 1e-10))

        elif self.mf_type == "bell":
            for j in range(self.n_features_):
                for k in range(n_mfs):
                    diff = X[:, j] - self.centers_[j, k]
                    mu[:, j, k] = 1.0 / (1.0 + (diff / (self.widths_[j, k] + 1e-10)) ** (2 * self.slopes_[j, k]) + 1e-10)

        elif self.mf_type == "triangular":
            for j in range(self.n_features_):
                for k in range(n_mfs):
                    l, c, r = self.lefts_[j, k], self.centers_[j, k], self.rights_[j, k]
                    x = X[:, j]
                    mu[:, j, k] = np.maximum(0, np.minimum(
                        (x - l) / (c - l + 1e-10),
                        (r - x) / (r - c + 1e-10)
                    ))

        # Clamp para evitar zeros totais
        mu = np.clip(mu, 1e-10, 1.0)
        return mu

    # ================================================================
    # L2 — Força de disparo (produto das pertinências)
    # ================================================================
    def _compute_firing(self, mu):
        """
        Calcula força de disparo de cada regra.
        Cada regra = combinação de uma MF por feature.

        Returns
        -------
        w : ndarray, shape (n_samples, n_rules)
        """
        n_samples = mu.shape[0]
        n_mfs = self.effective_n_mfs_

        # Gerar índices de regras (produto cartesiano)
        # Cada regra indexa uma MF por feature
        rule_indices = np.array(np.meshgrid(
            *[range(n_mfs) for _ in range(self.n_features_)]
        )).T.reshape(-1, self.n_features_)

        w = np.ones((n_samples, self.n_rules_))
        for i, rule in enumerate(rule_indices):
            for j, mf_idx in enumerate(rule):
                w[:, i] *= mu[:, j, mf_idx]

        return w

    # ================================================================
    # L3 — Normalização
    # ================================================================
    def _normalize_firing(self, w):
        """Normaliza as forças de disparo."""
        w_sum = w.sum(axis=1, keepdims=True) + 1e-10
        return w / w_sum

    # ================================================================
    # L4/L5 — Consequentes (LSE) e saída
    # ================================================================
    def _fit_consequents_lse(self, X, y, w_norm):
        """
        Ajusta consequentes lineares via Least Squares.
        f_i = [x1, x2, ..., xn, 1] · θ_i
        Saída = Σ w̄_i · f_i
        """
        n_samples = X.shape[0]

        # Montar matriz expandida: para cada regra, w̄_i * [x, 1]
        # Shape: (n_samples, n_rules * (n_features + 1))
        X_aug = np.hstack([X, np.ones((n_samples, 1))])  # (n_samples, n_features+1)
        cols = self.n_rules_ * (self.n_features_ + 1)
        A = np.zeros((n_samples, cols))

        for i in range(self.n_rules_):
            start = i * (self.n_features_ + 1)
            end = start + self.n_features_ + 1
            A[:, start:end] = w_norm[:, i:i+1] * X_aug

        # Ridge regression: θ = (AᵀA + λI)⁻¹ Aᵀy
        ATA = A.T @ A + self.lambda_reg * np.eye(cols)
        ATy = A.T @ y
        try:
            self.consequent_params_ = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            self.consequent_params_ = np.linalg.lstsq(ATA, ATy, rcond=None)[0]

    def _predict_with_params(self, X, w_norm):
        """Predição usando consequentes atuais."""
        n_samples = X.shape[0]
        X_aug = np.hstack([X, np.ones((n_samples, 1))])

        y_pred = np.zeros(n_samples)
        for i in range(self.n_rules_):
            start = i * (self.n_features_ + 1)
            end = start + self.n_features_ + 1
            f_i = X_aug @ self.consequent_params_[start:end]
            y_pred += w_norm[:, i] * f_i

        return y_pred

    # ================================================================
    # Backward — Atualizar premissas por gradiente
    # ================================================================
    def _update_premises(self, X, y, mu, w, w_norm, error):
        """Atualiza parâmetros das MFs por gradiente descendente."""
        n_samples = X.shape[0]
        n_mfs = self.effective_n_mfs_

        if self.mf_type != "gaussian":
            return  # Gradiente implementado apenas para Gaussian

        # Calcular f_i para cada regra
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        f_values = np.zeros((n_samples, self.n_rules_))
        for i in range(self.n_rules_):
            start = i * (self.n_features_ + 1)
            end = start + self.n_features_ + 1
            f_values[:, i] = X_aug @ self.consequent_params_[start:end]

        # y_pred
        y_pred = np.sum(w_norm * f_values, axis=1)

        # Gerar índices de regras
        rule_indices = np.array(np.meshgrid(
            *[range(n_mfs) for _ in range(self.n_features_)]
        )).T.reshape(-1, self.n_features_)

        w_sum = w.sum(axis=1, keepdims=True) + 1e-10

        # Gradientes para centros e sigmas
        for j in range(self.n_features_):
            for k in range(n_mfs):
                # Encontrar regras que usam MF k na feature j
                relevant_rules = [r for r, idx in enumerate(rule_indices) if idx[j] == k]

                if not relevant_rules:
                    continue

                # dmu/dc e dmu/dsigma para MF Gaussian
                diff = X[:, j] - self.centers_[j, k]
                sigma = self.sigmas_[j, k]

                dmu_dc = mu[:, j, k] * diff / (sigma ** 2 + 1e-10)
                dmu_ds = mu[:, j, k] * (diff ** 2) / (sigma ** 3 + 1e-10)

                # dE/dc e dE/ds via chain rule
                grad_c = 0.0
                grad_s = 0.0

                for r in relevant_rules:
                    # dw_r / dmu_{j,k} = w_r / mu_{j,k}
                    dw_dmu = w[:, r] / (mu[:, j, k] + 1e-10)

                    # d(w_norm_r) / dw_r
                    dwnorm_dw = (w_sum.ravel() - w[:, r]) / (w_sum.ravel() ** 2 + 1e-10)

                    # dE/dw_norm_r
                    dE_dwnorm = -(error) * (f_values[:, r] - y_pred)

                    chain = dE_dwnorm * dwnorm_dw * dw_dmu

                    grad_c += np.mean(chain * dmu_dc)
                    grad_s += np.mean(chain * dmu_ds)
                
                # Evitar gradientes explosivos (Gradient Clipping)
                grad_c = np.clip(grad_c, -100.0, 100.0)
                grad_s = np.clip(grad_s, -100.0, 100.0)
                
                self.centers_[j, k] -= self.learning_rate * grad_c
                self.sigmas_[j, k] -= self.learning_rate * grad_s
                
                # Manter centers em um intervalo razoável (X está em escala padrão)
                self.centers_[j, k] = np.clip(self.centers_[j, k], -100.0, 100.0)
                
                self.sigmas_[j, k] = max(self.sigmas_[j, k], 1e-6)


def get_model_name():
    return "ANFIS"


def get_optuna_search_space(trial, dataset_size=0):
    """Espaço de busca Optuna para ANFIS."""
    n_mfs = trial.suggest_int("n_mfs", 2, 5)
    mf_type = trial.suggest_categorical("mf_type", ["gaussian", "bell", "triangular"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_epochs = trial.suggest_int("n_epochs", 50, 300, step=50)
    lambda_reg = trial.suggest_float("lambda_reg", 1e-6, 1e-1, log=True)

    return {
        "n_mfs": n_mfs,
        "mf_type": mf_type,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "lambda_reg": lambda_reg,
    }


def create_model(params, random_state=None):
    """Cria pipeline StandardScaler + ANFISRegressor."""
    anfis = ANFISRegressor(
        n_mfs=params["n_mfs"],
        mf_type=params["mf_type"],
        learning_rate=params["learning_rate"],
        n_epochs=params["n_epochs"],
        lambda_reg=params["lambda_reg"],
        random_state=random_state,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", anfis),
    ])

    return pipeline
