"""
Configurações globais do projeto RNA & Neuro-Fuzzy.
"""
import os

# ============================================================
# Diretórios
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

# ============================================================
# Parâmetros do experimento
# ============================================================
N_RUNS = 21          # Execuções independentes por modelo/dataset
N_FOLDS = 5          # Folds da validação cruzada
TEST_SIZE = 0.2      # Fração do holdout test
N_TRIALS = 15        # Trials do Optuna por otimização
RANDOM_STATE = 42    # Seed base para split treino/teste (fixa)

# ============================================================
# Datasets Kaggle — [link_kaggle, caminho_local_csv]
# ============================================================
datasets = [
    ["vivek468/superstore-dataset-final",
     os.path.join(DATASET_DIR, "data1", "superstore.csv")],
    ["shubhambathwal/flight-price-prediction",
     os.path.join(DATASET_DIR, "data2", "flight_price.csv")],
    ["taeefnajib/used-car-price-prediction-dataset",
     os.path.join(DATASET_DIR, "data3", "used_car.csv")],
    ["aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster",
     os.path.join(DATASET_DIR, "data4", "obesity.csv")],
]

# ============================================================
# Mapeamento dataset → coluna target (regressão)
# ============================================================
TARGET_COLUMNS = {
    0: "Profit",      # Superstore
    1: "price",       # Flight price
    2: "price",       # Used car price
    3: "Weight",      # Obesity — peso corporal
}

# ============================================================
# Colunas a DROPAR por dataset (IDs, nomes, datas irrelevantes)
# ============================================================
DROP_COLUMNS = {
    0: ["Row ID", "Order ID", "Order Date", "Ship Date",
        "Customer ID", "Customer Name", "Product ID", "Product Name",
        "Country", "Postal Code"],
    1: [],             # Flight — usar todas (já limpo no Kaggle)
    2: [],             # Used car — decidido no preprocessing
    3: [],             # Obesity — decidido no preprocessing
}

# ============================================================
# Nomes dos datasets (para gráficos e relatórios)
# ============================================================
DATASET_NAMES = {
    0: "Superstore (Profit)",
    1: "Flight Price",
    2: "Used Car Price",
    3: "Obesity (Weight)",
}

# ============================================================
# Nomes dos modelos
# ============================================================
MODEL_NAMES = [
    "MLP",
    "RBF Network",
    "ANFIS",
    "FNN-FCM",
]
