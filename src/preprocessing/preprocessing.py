"""
Preprocessamento robusto dos 4 datasets para regressão.

Pipeline:
  1. Remoção de colunas irrelevantes (IDs, nomes, datas)
  2. Preprocessamento específico por dataset
  3. Remoção de duplicatas
  4. Tratamento de valores ausentes:
     - Colunas com >50% missing → removidas
#      - Colunas com <=5% missing -> linhas removidas
#      - Colunas com >5% e <=50% missing -> preenchidas (mediana/moda)
  5. One-Hot Encoding para categóricas nominais
  6. Normalização (MinMaxScaler) de toda a base
  7. Salvamento do CSV processado
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Adiciona src ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Limiar para decisão de missing values
_MISSING_DROP_COL_THRESHOLD = 0.50   # >50% missing → remove coluna
_MISSING_DROP_ROW_THRESHOLD = 0.05   # <=5% missing -> remove linhas
# Entre 5% e 50% -> preenche com mediana (numérico) ou moda (categórico)

# Limiar para one-hot vs label encoding
_ONEHOT_MAX_CARDINALITY = 100  # Se coluna tem ≤100 valores únicos → one-hot


def load_and_preprocess(dataset_index, force=False):
    """
    Carrega o dataset processado (CSV), ou processa do zero se necessário.

    Parameters
    ----------
    dataset_index : int — índice do dataset (0-3)
    force : bool — se True, reprocessa mesmo que o CSV processado já exista

    Returns
    -------
    X : np.ndarray — features normalizadas
    y : np.ndarray — target
    feature_names : list[str]
    """
    processed_path = _get_processed_path(dataset_index)

    if os.path.exists(processed_path) and not force:
        print(f"\n  Carregando dataset processado: {processed_path}")
        return _load_processed(dataset_index)
    else:
        print(f"\n  Processando dataset {dataset_index} do zero...")
        return preprocess_and_save(dataset_index)


def preprocess_and_save(dataset_index):
    """
    Executa todo o pipeline de preprocessamento e salva CSV processado.

    Returns
    -------
    X : np.ndarray — features normalizadas
    y : np.ndarray — target
    feature_names : list[str]
    """
    target_col = config.TARGET_COLUMNS[dataset_index]
    ds_name = config.DATASET_NAMES[dataset_index]
    caminho = config.datasets[dataset_index][1]

    # ---- 0. Carregar CSV raw ----
    try:
        df = pd.read_csv(caminho)
    except UnicodeDecodeError:
        df = pd.read_csv(caminho, encoding="latin1")

    print(f"\n{'='*60}")
    print(f"PREPROCESSAMENTO — Dataset {dataset_index}: {ds_name}")
    print(f"{'='*60}")
    print(f"  Shape original: {df.shape}")

    # ---- 1. Dropar colunas irrelevantes (config) ----
    cols_to_drop = config.DROP_COLUMNS.get(dataset_index, [])
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Colunas dropadas (config): {cols_to_drop}")

    # ---- 2. Preprocessamento específico por dataset ----
    if dataset_index == 0:
        df = _preprocess_superstore(df)
    elif dataset_index == 1:
        df = _preprocess_flight(df)
    elif dataset_index == 2:
        df = _preprocess_used_car(df)
    elif dataset_index == 3:
        df = _preprocess_obesity(df)

    # ---- 3. Remoção de duplicatas ----
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  Duplicatas removidas: {n_removed} ({n_removed/n_before*100:.1f}%)")

    # ---- 4. Tratamento de valores ausentes ----
    df = _handle_missing_values(df)

    # ---- 5. Verificar que target existe ----
    if target_col not in df.columns:
        raise ValueError(f"Coluna target '{target_col}' não encontrada. "
                         f"Colunas: {list(df.columns)}")

    # Remover linhas onde target é NaN
    n_before = len(df)
    df = df.dropna(subset=[target_col])
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"  Linhas com target NaN removidas: {n_removed}")

    # ---- 6. Separar target ----
    y = df[target_col].values.astype(np.float64)
    X_df = df.drop(columns=[target_col])

    # ---- 7. One-Hot Encoding para categóricas ----
    X_df = _encode_categoricals(X_df)

    # ---- 8. Normalização (MinMaxScaler) ----
    feature_names = list(X_df.columns)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df.values.astype(np.float64))

    # ---- 9. Salvar CSV processado ----
    processed_df = pd.DataFrame(X_scaled, columns=feature_names)
    processed_df[target_col] = y
    processed_path = _get_processed_path(dataset_index)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    processed_df.to_csv(processed_path, index=False)

    print(f"\n  Shape final: X={X_scaled.shape}, y={y.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Target '{target_col}' — min: {y.min():.2f}, max: {y.max():.2f}, "
          f"mean: {y.mean():.2f}, std: {y.std():.2f}")
    print(f"  CSV processado salvo: {processed_path}")
    print(f"{'='*60}")

    return X_scaled, y, feature_names


def _load_processed(dataset_index):
    """Carrega dataset já processado do CSV."""
    processed_path = _get_processed_path(dataset_index)
    target_col = config.TARGET_COLUMNS[dataset_index]

    df = pd.read_csv(processed_path)

    y = df[target_col].values.astype(np.float64)
    X_df = df.drop(columns=[target_col])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float64)

    print(f"  Shape: X={X.shape}, y={y.shape} | Features: {len(feature_names)}")
    return X, y, feature_names


def _get_processed_path(dataset_index):
    """Retorna o caminho do CSV processado."""
    names = {0: "superstore", 1: "flight_price",
             2: "used_car", 3: "obesity"}
    return os.path.join(config.PROCESSED_DIR,
                        f"{names[dataset_index]}_processed.csv")


def check_processed_exists(dataset_index):
    """Verifica se o CSV processado já existe."""
    return os.path.exists(_get_processed_path(dataset_index))


# ============================================================
# Tratamento de valores ausentes
# ============================================================

def _handle_missing_values(df):
    """
    Estratégia de tratamento de missing values:
    - Colunas com >50% missing → removidas
    - Colunas com ≤5% missing → linhas com missing removidas
    - Colunas com >5% e ≤50% missing → preenchidas
    """
    n_rows = len(df)
    missing_report = []

    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        pct = n_missing / n_rows
        missing_report.append((col, n_missing, pct))

    if not missing_report:
        print("  Valores ausentes: nenhum encontrado")
        return df

    print(f"\n  Valores ausentes encontrados ({len(missing_report)} colunas):")

    # Fase 1: Remover colunas com >50% missing
    cols_to_remove = []
    for col, n_missing, pct in missing_report:
        if pct > _MISSING_DROP_COL_THRESHOLD:
            cols_to_remove.append(col)
            print(f"    {col}: {n_missing} ({pct*100:.1f}%) -> COLUNA REMOVIDA")

    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)

    # Fase 2: Para colunas restantes com missing
    rows_to_drop_mask = pd.Series(False, index=df.index)

    for col, n_missing, pct in missing_report:
        if col in cols_to_remove:
            continue
        if col not in df.columns:
            continue

        if pct <= _MISSING_DROP_ROW_THRESHOLD:
            # <=5% -> marcar linhas para remoção
            rows_to_drop_mask |= df[col].isnull()
            print(f"    {col}: {n_missing} ({pct*100:.1f}%) -> linhas removidas")
        else:
            # >5% e <=50% -> preencher
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                fill_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown"
                df[col] = df[col].fillna(fill_val)
            print(f"    {col}: {n_missing} ({pct*100:.1f}%) -> preenchido (mediana/moda)")

    # Aplicar remoção de linhas
    n_dropped = rows_to_drop_mask.sum()
    if n_dropped > 0:
        df = df[~rows_to_drop_mask].reset_index(drop=True)
        print(f"  Total de linhas removidas (missing <=5%): {n_dropped}")

    return df


# ============================================================
# Preprocessamento específico por dataset
# ============================================================

def _preprocess_superstore(df):
    """Superstore: Sales, Quantity, Discount, Profit + categóricas."""
    for col in ["Country", "Customer Name", "Product Name"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _preprocess_flight(df):
    """Flight Price: remover código do voo (ID) e limpar/converter price."""
    if "flight" in df.columns:
        df = df.drop(columns=["flight"])
    unnamed_cols = [c for c in df.columns if "Unnamed" in c]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    
    # Limpar coluna 'price' (pode ter vírgulas, ex: '25,612')
    if "price" in df.columns:
        df["price"] = df["price"].astype(str).str.replace(",", "", regex=False)
        df["price"] = df["price"].str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def _preprocess_used_car(df):
    """Used Car Price: limpeza e engenharia de atributos avançada."""
    # 1. Dropar apenas 'model' (texto livre muito longo de alta cardinalidade)
    if "model" in df.columns:
        df = df.drop(columns=["model"])
        
    unnamed_cols = [c for c in df.columns if "Unnamed" in c]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # 2. Limpar 'milage' (ex: "51,000 mi.")
    if "milage" in df.columns:
        df["milage"] = df["milage"].astype(str).str.replace("mi.", "", regex=False)
        df["milage"] = df["milage"].str.replace(",", "", regex=False)
        df["milage"] = df["milage"].str.strip()
        df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

    # 3. Limpar 'price' (ex: "$10,300")
    if "price" in df.columns:
        df["price"] = df["price"].astype(str).str.replace("$", "", regex=False)
        df["price"] = df["price"].str.replace(",", "", regex=False)
        df["price"] = df["price"].str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 4. Tratar 'clean_title' (NaN -> "No", depois one-hot)
    if "clean_title" in df.columns:
        df["clean_title"] = df["clean_title"].fillna("No").astype(str).str.strip()

    # 5. Tratar 'accident' (virar duas colunas explicitas)
    if "accident" in df.columns:
        acc_str = df["accident"].fillna("unknown").astype(str).str.lower()
        df["accident_reported"] = acc_str.str.contains("accident|damage").astype(np.float64)
        df["accident_none"] = acc_str.str.contains("none").astype(np.float64)
        df = df.drop(columns=["accident"])

    # 6. Simplificar e manter 'ext_col' e 'int_col' para evitar alta cardinalidade
    def simplify_color(color_series):
        colors_map = {
            "black": "black", "white": "white", "blue": "blue", "red": "red",
            "silver": "silver", "gray": "gray", "grey": "gray", "green": "green",
            "brown": "brown", "beige": "beige", "yellow": "yellow", "gold": "gold",
            "orange": "orange", "purple": "purple", "charcoal": "gray", "bronze": "brown",
            "ebony": "black", "tan": "brown"
        }
        simplified = []
        for val in color_series.fillna("other").astype(str).str.lower():
            found = False
            for k, v in colors_map.items():
                if k in val:
                    simplified.append(v)
                    found = True
                    break
            if not found:
                simplified.append("other")
        return simplified

    if "ext_col" in df.columns:
        df["ext_col"] = simplify_color(df["ext_col"])
    if "int_col" in df.columns:
        df["int_col"] = simplify_color(df["int_col"])

    # 7. Engenharia de Atributos para 'engine' e 'transmission'
    # Extrair Horsepower (HP)
    if "engine" in df.columns:
        hp_extract = df["engine"].astype(str).str.extract(r"(\d+\.?\d*)\s*HP", expand=False)
        df["engine_hp"] = pd.to_numeric(hp_extract, errors="coerce")
        # Extrair Litros (L)
        l_extract = df["engine"].astype(str).str.extract(r"(\d+\.?\d*)\s*(?:L|Liter)", expand=False)
        df["engine_liters"] = pd.to_numeric(l_extract, errors="coerce")
        
        # Se for elétrico, cilindrada é 0
        is_electric = df["engine"].astype(str).str.lower().str.contains("electric")
        df.loc[is_electric, "engine_liters"] = 0.0
        
        # Dropar a coluna original complexa
        df = df.drop(columns=["engine"])

    if "transmission" in df.columns:
        trans_str = df["transmission"].fillna("other").astype(str).str.lower()
        df["trans_automatic"] = trans_str.str.contains("automatic|a/t|cvt|dual shift|at|auto|speed a/t").astype(np.float64)
        df["trans_manual"] = trans_str.str.contains("manual|m/t|mt|speed m/t").astype(np.float64)
        df = df.drop(columns=["transmission"])

    # 8. Converter model_year para idade do carro (age) - facilita o aprendizado do modelo
    if "model_year" in df.columns:
        df["car_age"] = 2026 - df["model_year"]
        df = df.drop(columns=["model_year"])
    return df


def _preprocess_obesity(df):
    """Obesity: usar Weight como target, dropar NObeyesdad."""
    if "NObeyesdad" in df.columns:
        df = df.drop(columns=["NObeyesdad"])
    return df


# ============================================================
# Encoding de variáveis categóricas
# ============================================================

def _encode_categoricals(df):
    """
    One-Hot Encoding para categóricas com baixa cardinalidade (≤15 valores).
    Label Encoding para categóricas com alta cardinalidade (>15 valores).
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        return df

    onehot_cols = []
    label_cols = []

    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= _ONEHOT_MAX_CARDINALITY:
            onehot_cols.append(col)
        else:
            label_cols.append(col)

    # One-Hot Encoding
    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=np.float64)
        print(f"  One-Hot Encoding: {onehot_cols}")

    # Label Encoding (alta cardinalidade)
    if label_cols:
        le = LabelEncoder()
        for col in label_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Label Encoding (alta cardinalidade): {label_cols}")

    return df


# ============================================================
# Utilitário: preprocessar todos os datasets
# ============================================================

def preprocess_all(force=False):
    """Preprocessa todos os 4 datasets e salva CSVs processados."""
    print("\n" + "#" * 60)
    print("# PREPROCESSAMENTO DE TODOS OS DATASETS")
    print("#" * 60)

    for i in range(len(config.datasets)):
        try:
            if check_processed_exists(i) and not force:
                print(f"\n  Dataset {i} ({config.DATASET_NAMES[i]}): "
                      f"já processado. Use force=True para reprocessar.")
            else:
                preprocess_and_save(i)
        except Exception as e:
            print(f"\n  ERRO no dataset {i}: {e}")
            import traceback
            traceback.print_exc()

    print("\n  Preprocessamento concluído!")
