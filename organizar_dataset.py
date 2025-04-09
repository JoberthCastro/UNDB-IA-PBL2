import os
import shutil
import random
from pathlib import Path
import logging
import requests
from tqdm import tqdm
import zipfile

def configurar_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def baixar_arquivo(url, nome_arquivo):
    """Baixa um arquivo da internet mostrando o progresso."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(nome_arquivo, 'wb') as file, tqdm(
        desc=nome_arquivo,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def baixar_dataset():
    """Baixa o dataset de um link direto."""
    logger = logging.getLogger(__name__)
    logger.info("Baixando dataset...")
    
    try:
        # URL do dataset (substitua pela URL correta)
        url = "https://storage.googleapis.com/kaggle-data-sets/1217390/2032566/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240407%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240407T102646Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=94a6b0b614a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6"
        
        # Nome do arquivo zip
        zip_file = "dataset.zip"
        
        # Baixar o arquivo
        logger.info("Baixando arquivo zip...")
        baixar_arquivo(url, zip_file)
        
        # Extrair o arquivo
        logger.info("Extraindo arquivo zip...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Remover o arquivo zip
        os.remove(zip_file)
        
        dataset_path = "."
        logger.info(f"Dataset baixado e extraído com sucesso em: {dataset_path}")
        return dataset_path
    except Exception as e:
        logger.error(f"Erro ao baixar dataset: {str(e)}")
        raise

def criar_estrutura_diretorios(destino):
    """Cria a estrutura de diretórios necessária."""
    for split in ['train', 'val', 'test']:
        for classe in ['covid', 'normal']:
            path = os.path.join(destino, split, classe)
            os.makedirs(path, exist_ok=True)
            logging.info(f"Criado diretório: {path}")

def contar_imagens(diretorio):
    """Conta o número de imagens em um diretório."""
    return len([f for f in os.listdir(diretorio) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def organizar_dataset(origem, destino, proporcoes=(0.7, 0.15, 0.15)):
    """
    Organiza o dataset em treino, validação e teste
    
    Args:
        origem: Caminho para a pasta com as imagens originais
        destino: Caminho para criar a estrutura do dataset
        proporcoes: Tupla com as proporções (treino, validação, teste)
    """
    logger = configurar_logger()
    
    # Verificar se as proporções somam 1
    if sum(proporcoes) != 1:
        raise ValueError("As proporções devem somar 1")
    
    # Criar estrutura de diretórios
    criar_estrutura_diretorios(destino)
    
    # Verificar diretórios de origem
    diretorios_origem = {
        'covid': os.path.join(origem, "COVID-19_Radiography_Dataset", "COVID"),
        'normal': os.path.join(origem, "COVID-19_Radiography_Dataset", "Normal")
    }
    
    for classe, dir_path in diretorios_origem.items():
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Diretório não encontrado: {dir_path}")
        
        # Listar imagens
        imagens = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(imagens)
        
        # Calcular divisões
        total = len(imagens)
        n_treino = int(total * proporcoes[0])
        n_val = int(total * proporcoes[1])
        
        logger.info(f"\nProcessando classe: {classe}")
        logger.info(f"Total de imagens: {total}")
        logger.info(f"Treino: {n_treino}")
        logger.info(f"Validação: {n_val}")
        logger.info(f"Teste: {total - n_treino - n_val}")
        
        # Dividir e copiar imagens
        divisoes = {
            'train': imagens[:n_treino],
            'val': imagens[n_treino:n_treino + n_val],
            'test': imagens[n_treino + n_val:]
        }
        
        for split, imgs in divisoes.items():
            dest_dir = os.path.join(destino, split, classe)
            for img in imgs:
                shutil.copy2(
                    os.path.join(dir_path, img),
                    os.path.join(dest_dir, img)
                )
    
    # Verificar e reportar resultados
    for split in ['train', 'val', 'test']:
        logger.info(f"\nEstatísticas do conjunto {split}:")
        for classe in ['covid', 'normal']:
            path = os.path.join(destino, split, classe)
            n_imagens = contar_imagens(path)
            logger.info(f"{classe}: {n_imagens} imagens")

if __name__ == "__main__":
    try:
        # Baixar dataset
        dataset_path = baixar_dataset()
        
        # Organizar dataset
        organizar_dataset(dataset_path, "dataset")
        print("\nDataset organizado com sucesso!")
    except Exception as e:
        print(f"\nErro ao organizar dataset: {str(e)}") 