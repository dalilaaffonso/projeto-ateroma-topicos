import os
import shutil
from sklearn.model_selection import train_test_split

def criar_estrutura(base_dir):
    """Cria as pastas e subpastas automaticamente se não existirem."""
    splits = ['treino', 'validacao', 'teste']
    classes = ['com_ateroma', 'sem_ateroma']
    
    for split in splits:
        for classe in classes:
            os.makedirs(os.path.join(base_dir, split, classe), exist_ok=True)

def obter_caminhos_imagens(diretorio_base):
    """Entra em todas as subpastas (como GV e dataset público) e coleta os caminhos das imagens."""
    caminhos = []
    for root, dirs, files in os.walk(diretorio_base):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                caminhos.append(os.path.join(root, f))
    return caminhos

def dividir_e_copiar(origem_com, origem_sem, base_dir):
    criar_estrutura(base_dir)
    
    # Coleta todas as imagens, independentemente de estarem em subpastas
    imgs_com = obter_caminhos_imagens(origem_com)
    imgs_sem = obter_caminhos_imagens(origem_sem)
    
    # Divisão 60/20/20 para "Com Ateroma"
    com_treino, com_temp = train_test_split(imgs_com, test_size=0.4, random_state=42)
    com_val, com_teste = train_test_split(com_temp, test_size=0.5, random_state=42)
    
    # Divisão 60/20/20 para "Sem Ateroma"
    sem_treino, sem_temp = train_test_split(imgs_sem, test_size=0.4, random_state=42)
    sem_val, sem_teste = train_test_split(sem_temp, test_size=0.5, random_state=42)
    
    # Função auxiliar para copiar
    def copiar(caminhos_origem, split_nome, classe_nome):
        for caminho in caminhos_origem:
            nome_arquivo = os.path.basename(caminho)
            destino = os.path.join(base_dir, split_nome, classe_nome, nome_arquivo)
            shutil.copy(caminho, destino)
            
    # Executando as cópias
    copiar(com_treino, 'treino', 'com_ateroma')
    copiar(com_val, 'validacao', 'com_ateroma')
    copiar(com_teste, 'teste', 'com_ateroma')
    
    copiar(sem_treino, 'treino', 'sem_ateroma')
    copiar(sem_val, 'validacao', 'sem_ateroma')
    copiar(sem_teste, 'teste', 'sem_ateroma')
    
    print(f"Divisão concluída! Foram processadas {len(imgs_com)} imagens com ateroma e {len(imgs_sem)} sem ateroma.")

# Altere os caminhos abaixo de acordo com o seu computador
caminho_com_cac = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Ateroma_1\Com_cac"
caminho_sem_cac = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Ateroma_1\Sem_CAC"
pasta_destino = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Dividido"

dividir_e_copiar(caminho_com_cac, caminho_sem_cac, pasta_destino)