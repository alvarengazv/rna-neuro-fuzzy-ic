import kagglehub
import os
import shutil
import pandas as pd
import config

def clear_terminal():
    for _ in range(100):
        print()
    os.system('cls' if os.name == 'nt' else 'clear')    

def check_dataset():
    for _, caminho_local in config.datasets:
        if not os.path.exists(caminho_local):
            return False
            
    print("Todos os datasets já foram baixados e estão prontos.")
    return True

def get_data():
    for link_kaggle, caminho_local in config.datasets:
        print(f"Processando: {link_kaggle}...")
        
        # Baixa pro cache
        cache_path = kagglehub.dataset_download(link_kaggle)
        
        # Pega apenas a parte da pasta (ex: "dataset/data1") e cria se não existir
        caminho_pasta = os.path.dirname(caminho_local)
        os.makedirs(caminho_pasta, exist_ok=True)
        
        # Procura o CSV no cache e copia para o caminho_local
        for arquivo in os.listdir(cache_path):
            if arquivo.endswith(".csv"):
                caminho_origem = os.path.join(cache_path, arquivo)
                shutil.copy2(caminho_origem, caminho_local)
                print(f"Salvo em: {caminho_local}\n")
                break

def read_datasets():
    df_list = []
    for _, caminho_local in config.datasets:
        print(f"Lendo: {caminho_local}...")
        try:
            df = pd.read_csv(caminho_local)
        except UnicodeDecodeError:
            print(f"  -> Aviso: Erro de UTF-8 detectado. Lendo com 'latin1'...")
            df = pd.read_csv(caminho_local, encoding='latin1')
            
        df_list.append(df)
    
    return df_list

def main():
   if not check_dataset():
       get_data()
   
   datasets = read_datasets()

if __name__ == "__main__":
    main()