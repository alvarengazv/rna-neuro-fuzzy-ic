"""Quick test: verifica o pipeline de preprocessamento."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import config
from preprocessing.preprocessing import (
    _handle_missing_values, _encode_categoricals,
    _get_processed_path, check_processed_exists
)

# Criar dataset sintético com problemas reais
np.random.seed(42)
n = 100
df = pd.DataFrame({
    "num1": np.random.randn(n),
    "num2": np.random.randn(n),
    "cat_low": np.random.choice(["A", "B", "C"], n),          # One-hot (3 vals)
    "cat_high": np.random.choice([f"v{i}" for i in range(20)], n),  # Label (20 vals)
    "mostly_missing": np.where(np.random.rand(n) < 0.6, np.nan, 1.0),  # >50% → drop col
    "few_missing": np.where(np.random.rand(n) < 0.03, np.nan, 1.0),    # ≤5% → drop rows
    "some_missing": np.where(np.random.rand(n) < 0.15, np.nan, 1.0),   # 15% → fill median
    "target": np.random.randn(n) * 10 + 50,
})

# Adicionar duplicatas
df = pd.concat([df, df.head(5)], ignore_index=True)
print(f"Shape com duplicatas: {df.shape}")

# Testar tratamento de missing
print("\n--- Tratamento de missing ---")
df_clean = _handle_missing_values(df.copy())
print(f"Shape após missing: {df_clean.shape}")
assert "mostly_missing" not in df_clean.columns, "Coluna >50% deveria ter sido removida"

# Testar duplicatas
df_clean = df_clean.drop_duplicates()
print(f"Shape após duplicatas: {df_clean.shape}")

# Testar encoding
print("\n--- Encoding ---")
y = df_clean["target"]
X = df_clean.drop(columns=["target"])
X = _encode_categoricals(X.copy())
print(f"Shape após encoding: {X.shape}")
print(f"Colunas: {list(X.columns)}")

# Verificar que one-hot criou colunas certas
assert any("cat_low" in c for c in X.columns), "One-hot deveria ter criado colunas cat_low_*"

# Verificar paths
for i in range(4):
    path = _get_processed_path(i)
    print(f"Dataset {i} processed path: {path}")
    print(f"  Exists: {check_processed_exists(i)}")

print("\n✓ TODOS OS TESTES PASSARAM!")
