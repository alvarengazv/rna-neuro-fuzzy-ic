"""
Ponto de entrada principal do projeto RNA & Neuro-Fuzzy.
Menu interativo para controlar execução: download, EDA, preprocessamento, experimentos.
"""
import os
import sys
import shutil

sys.stdout.reconfigure(encoding='utf-8')

# Adiciona src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def check_dataset():
    """Verifica se todos os datasets já foram baixados."""
    for _, caminho_local in config.datasets:
        if not os.path.exists(caminho_local):
            return False
    return True


def get_data():
    """Baixa os datasets do Kaggle via kagglehub."""
    import kagglehub

    for link_kaggle, caminho_local in config.datasets:
        if os.path.exists(caminho_local):
            print(f"  ✓ Já existe: {caminho_local}")
            continue

        print(f"  Baixando: {link_kaggle}...")
        cache_path = kagglehub.dataset_download(link_kaggle)

        caminho_pasta = os.path.dirname(caminho_local)
        os.makedirs(caminho_pasta, exist_ok=True)

        for arquivo in os.listdir(cache_path):
            if arquivo.endswith(".csv"):
                caminho_origem = os.path.join(cache_path, arquivo)
                shutil.copy2(caminho_origem, caminho_local)
                print(f"  ✓ Salvo em: {caminho_local}")
                break


def print_header():
    """Imprime o cabeçalho do programa."""
    print("\n" + "=" * 70)
    print("  🧠 RNA & Neuro-Fuzzy — Experimentos de Regressão")
    print("  CEFET-MG — Inteligência Computacional — 2026/1")
    print("=" * 70)


def print_menu():
    """Imprime o menu principal."""
    print("\n" + "-" * 50)
    print("  MENU PRINCIPAL")
    print("-" * 50)
    print("  1. Baixar datasets do Kaggle")
    print("  2. Executar EDA (Análise Exploratória)")
    print("  3. Executar Preprocessamento")
    print("  4. Executar experimento (escolher modelo e dataset)")
    print("  5. Executar TODOS os experimentos")
    print("  6. Computador 1 (Flight Price - ANFIS apenas)")
    print("  7. Computador 2 (Flight Price - FNN-FCM apenas)")
    print("  8. Computador 3 (Flight Price - MLP/RBF apenas)")
    print("  9. Computador 4 (Superstore, Used Car, Obesity - Todos)")
    print("  0. Sair")
    print("-" * 50)


def choose_datasets():
    """Menu para escolher quais datasets usar."""
    print("\n  Datasets disponíveis:")
    for i, name in config.DATASET_NAMES.items():
        from preprocessing.preprocessing import check_processed_exists
        status = "✓ processado" if check_processed_exists(i) else "○ não processado"
        print(f"    {i}. {name}  [{status}]")
    print(f"    A. Todos os datasets")

    choice = input("\n  Escolha (0-3, ou A para todos): ").strip().upper()

    if choice == "A":
        return list(range(len(config.datasets)))
    elif choice.isdigit() and 0 <= int(choice) <= 3:
        return [int(choice)]
    else:
        print("  Opção inválida.")
        return None


def choose_models():
    """Menu para escolher quais modelos usar."""
    print("\n  Modelos disponíveis:")
    for i, name in enumerate(config.MODEL_NAMES):
        print(f"    {i}. {name}")
    print(f"    A. Todos os modelos")

    choice = input("\n  Escolha (0-3, ou A para todos): ").strip().upper()

    if choice == "A":
        return list(range(len(config.MODEL_NAMES)))
    elif choice.isdigit() and 0 <= int(choice) <= 3:
        return [int(choice)]
    else:
        print("  Opção inválida.")
        return None


def choose_params():
    """Permite ao usuário customizar parâmetros de execução."""
    print(f"\n  Parâmetros atuais:")
    print(f"    Trials Optuna: {config.N_TRIALS}")
    print(f"    Runs independentes: {config.N_RUNS}")
    print(f"    Folds CV: {config.N_FOLDS}")

    custom = input("\n  Deseja alterar? (s/N): ").strip().lower()

    n_trials = config.N_TRIALS
    n_runs = config.N_RUNS
    n_folds = config.N_FOLDS

    if custom == "s":
        try:
            val = input(f"    Trials Optuna [{n_trials}]: ").strip()
            if val:
                n_trials = int(val)
            val = input(f"    Runs independentes [{n_runs}]: ").strip()
            if val:
                n_runs = int(val)
            val = input(f"    Folds CV [{n_folds}]: ").strip()
            if val:
                n_folds = int(val)
        except ValueError:
            print("  Valor inválido, usando padrão.")

    return n_trials, n_runs, n_folds


def action_download():
    """Ação: baixar datasets."""
    print("\n  Verificando e baixando datasets...")
    if check_dataset():
        print("  ✓ Todos os datasets já existem.")
        redownload = input("  Deseja re-baixar? (s/N): ").strip().lower()
        if redownload != "s":
            return
    get_data()
    print("  ✓ Download concluído!")


def action_eda():
    """Ação: executar análise exploratória."""
    datasets = choose_datasets()
    if datasets is None:
        return

    from eda.eda import run_eda
    for ds_idx in datasets:
        try:
            run_eda(ds_idx)
        except Exception as e:
            print(f"  ERRO no EDA do dataset {ds_idx}: {e}")


def action_preprocess():
    """Ação: executar preprocessamento."""
    datasets = choose_datasets()
    if datasets is None:
        return

    from preprocessing.preprocessing import preprocess_and_save, check_processed_exists

    force = False
    # Verificar se algum já foi processado
    any_processed = any(check_processed_exists(i) for i in datasets)
    if any_processed:
        force_input = input("\n  Alguns datasets já foram processados. "
                           "Reprocessar? (s/N): ").strip().lower()
        force = (force_input == "s")

    for ds_idx in datasets:
        try:
            if check_processed_exists(ds_idx) and not force:
                print(f"\n  Dataset {ds_idx} ({config.DATASET_NAMES[ds_idx]}): "
                      f"já processado. Pulando.")
            else:
                preprocess_and_save(ds_idx)
        except Exception as e:
            print(f"\n  ERRO no preprocessamento do dataset {ds_idx}: {e}")
            import traceback
            traceback.print_exc()

    print("\n  ✓ Preprocessamento concluído!")


def action_experiment():
    """Ação: executar experimento com escolha de modelo e dataset."""
    # Verificar se datasets existem
    if not check_dataset():
        print("\n  ⚠ Datasets não encontrados. Execute o download primeiro (opção 1).")
        return

    # Escolher datasets
    datasets = choose_datasets()
    if datasets is None:
        return

    # Verificar preprocessamento
    from preprocessing.preprocessing import check_processed_exists, preprocess_and_save
    for ds_idx in datasets:
        if not check_processed_exists(ds_idx):
            print(f"\n  ⚠ Dataset {ds_idx} ({config.DATASET_NAMES[ds_idx]}) "
                  f"não foi preprocessado.")
            auto = input("  Preprocessar automaticamente? (S/n): ").strip().lower()
            if auto != "n":
                preprocess_and_save(ds_idx)
            else:
                print("  Abortando.")
                return

    # Escolher modelos
    models = choose_models()
    if models is None:
        return

    # Parâmetros
    n_trials, n_runs, n_folds = choose_params()

    # Confirmar
    ds_names = [config.DATASET_NAMES[i] for i in datasets]
    mod_names = [config.MODEL_NAMES[i] for i in models]
    print(f"\n  Resumo da execução:")
    print(f"    Datasets: {ds_names}")
    print(f"    Modelos: {mod_names}")
    print(f"    Trials: {n_trials} | Runs: {n_runs} | Folds: {n_folds}")
    print(f"    Estimativa: {len(datasets) * len(models)} combinações")

    confirm = input("\n  Confirmar execução? (S/n): ").strip().lower()
    if confirm == "n":
        print("  Execução cancelada.")
        return

    # Executar
    from experiments import run_all_experiments
    results = run_all_experiments(
        dataset_indices=datasets,
        model_indices=models,
        n_trials=n_trials,
        n_runs=n_runs,
        n_folds=n_folds,
    )

    print("\n  ✓ Experimentos concluídos! Resultados em:", config.RESULTS_DIR)


def action_run_all():
    """Ação: executar todos os experimentos (todos os modelos e datasets)."""
    if not check_dataset():
        print("\n  ⚠ Datasets não encontrados. Execute o download primeiro (opção 1).")
        return

    # Preprocessar todos que ainda não foram
    from preprocessing.preprocessing import check_processed_exists, preprocess_and_save
    for i in range(len(config.datasets)):
        if not check_processed_exists(i):
            print(f"\n  Preprocessando dataset {i} ({config.DATASET_NAMES[i]})...")
            preprocess_and_save(i)

    # Parâmetros
    n_trials, n_runs, n_folds = choose_params()

    total = len(config.datasets) * len(config.MODEL_NAMES)
    print(f"\n  Execução COMPLETA:")
    print(f"    {len(config.datasets)} datasets × {len(config.MODEL_NAMES)} modelos "
          f"= {total} combinações")
    print(f"    {n_runs} runs cada = {total * n_runs} execuções totais")

    confirm = input("\n  Confirmar? (S/n): ").strip().lower()
    if confirm == "n":
        print("  Execução cancelada.")
        return

    from experiments import run_all_experiments
    results = run_all_experiments(
        n_trials=n_trials, n_runs=n_runs, n_folds=n_folds
    )

    print("\n  ✓ Todos os experimentos concluídos! Resultados em:", config.RESULTS_DIR)


def run_pc_experiment(datasets, models, pc_name):
    """Auxiliar: executa o experimento configurado para uma máquina específica."""
    if not check_dataset():
        print(f"\n  ⚠ Datasets não encontrados. Execute o download primeiro (opção 1).")
        return

    # Verificar preprocessamento
    from preprocessing.preprocessing import check_processed_exists, preprocess_and_save
    for ds_idx in datasets:
        if not check_processed_exists(ds_idx):
            print(f"\n  ⚠ Dataset {ds_idx} ({config.DATASET_NAMES[ds_idx]}) não foi preprocessado.")
            auto = input("  Preprocessar automaticamente? (S/n): ").strip().lower()
            if auto != "n":
                preprocess_and_save(ds_idx)
            else:
                print("  Abortando.")
                return

    # Parâmetros
    n_trials, n_runs, n_folds = choose_params()

    # Confirmar
    ds_names = [config.DATASET_NAMES[i] for i in datasets]
    mod_names = [config.MODEL_NAMES[i] for i in models]
    print(f"\n  Resumo da execução ({pc_name}):")
    print(f"    Datasets: {ds_names}")
    print(f"    Modelos: {mod_names}")
    print(f"    Trials: {n_trials} | Runs: {n_runs} | Folds: {n_folds}")
    print(f"    Estimativa: {len(datasets) * len(models)} combinações")

    confirm = input("\n  Confirmar execução? (S/n): ").strip().lower()
    if confirm == "n":
        print("  Execução cancelada.")
        return

    # Executar
    from experiments import run_all_experiments
    run_all_experiments(
        dataset_indices=datasets,
        model_indices=models,
        n_trials=n_trials,
        n_runs=n_runs,
        n_folds=n_folds,
    )
    print(f"\n  ✓ Experimentos do {pc_name} concluídos! Resultados em: {config.RESULTS_DIR}")


def action_pc1():
    run_pc_experiment([1], [2], "Computador 1")


def action_pc2():
    run_pc_experiment([1], [3], "Computador 2")


def action_pc3():
    run_pc_experiment([1], [0, 1], "Computador 3")


def action_pc4():
    run_pc_experiment([0, 2, 3], [0, 1, 2, 3], "Computador 4")


def main():
    """Loop principal do menu interativo."""
    print_header()

    while True:
        print_menu()
        choice = input("  Escolha uma opção: ").strip()

        if choice == "1":
            action_download()
        elif choice == "2":
            action_eda()
        elif choice == "3":
            action_preprocess()
        elif choice == "4":
            action_experiment()
        elif choice == "5":
            action_run_all()
        elif choice == "6":
            action_pc1()
        elif choice == "7":
            action_pc2()
        elif choice == "8":
            action_pc3()
        elif choice == "9":
            action_pc4()
        elif choice == "0":
            print("\n  Até logo! 👋")
            break
        else:
            print("  Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()