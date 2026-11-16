import os
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def criar_estrutura_final(base_dir):
    """Cria a estrutura de pastas considerando o paciente como centro do diagnóstico."""
    splits = ['treino', 'validacao', 'teste']
    classes = ['com_ateroma', 'sem_ateroma']
    
    for split in splits:
        for classe in classes:
            os.makedirs(os.path.join(base_dir, split, classe), exist_ok=True)

def processar_e_dividir_pacientes(pasta_imagens_originais, arquivo_labels_csv, pasta_destino, tamanho_corte=512):
    """
    Recorta, espelha, nomeia e divide os dados focando no PACIENTE.
    Necessita de um DataFrame ou CSV com os rótulos do paciente.
    """
    import pandas as pd
    
    criar_estrutura_final(pasta_destino)
    
    # Carrega a planilha de anotações que geramos no Streamlit 
    # Ela tem as colunas: id_img, lesao_esquerda, lesao_direita
    df = pd.read_csv(arquivo_labels_csv, encoding='latin1', sep=None, engine='python')
    
    # Vamos dividir os PACIENTES em 60/20/20, não as imagens soltas
    pacientes = df['id_img'].unique().tolist()
    treino_pac, temp_pac = train_test_split(pacientes, test_size=0.4, random_state=42)
    val_pac, teste_pac = train_test_split(temp_pac, test_size=0.5, random_state=42)
    
    # Dicionário rápido para saber em qual split o paciente caiu
    mapa_splits = {p: 'treino' for p in treino_pac}
    mapa_splits.update({p: 'validacao' for p in val_pac})
    mapa_splits.update({p: 'teste' for p in teste_pac})
    
    processados = 0
    
    for index, row in df.iterrows():
        nome_arquivo = row['id_img']
        caminho_img = os.path.join(pasta_imagens_originais, row['subdir'], nome_arquivo)
        
        if not os.path.exists(caminho_img):
            print(f"Caminho falhou: {caminho_img}")
            continue
            
        img = cv2.imread(caminho_img)
        if img is None:
            continue
            
        altura, largura, _ = img.shape
        # Se a imagem for muito pequena, nós damos um "zoom" matemático nela 
        # (interpolação cúbica) até atingir o tamanho mínimo seguro antes de cortar.
        if altura < tamanho_corte or largura < tamanho_corte * 2:
            nova_altura = max(altura, tamanho_corte)
            nova_largura = max(largura, tamanho_corte * 2)
            img = cv2.resize(img, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)
            altura, largura, _ = img.shape  # Atualiza as dimensões após o zoom
            print(f"Redimensionamento aplicado na imagem {nome_arquivo} para extrair ROI.")
            
        # 1. Recortes Geométricos (Canto Inferior)
        # Esquerdo
        recorte_esq = img[altura - tamanho_corte:altura, 0:tamanho_corte]
        # Direito
        recorte_dir = img[altura - tamanho_corte:altura, largura - tamanho_corte:largura]
        
        # 2. Espelhamento Anatômico (Horizontal Flip)
        recorte_dir_espelhado = cv2.flip(recorte_dir, 1)
        
        # 3. Mapeamento de Classes e Salvamento
        split_atual = mapa_splits[nome_arquivo]
        nome_base = os.path.splitext(nome_arquivo)[0]
        
        # Salva Lado Esquerdo
        classe_esq = 'com_ateroma' if row['lesao_esquerda'] else 'sem_ateroma'
        caminho_salvar_esq = os.path.join(pasta_destino, split_atual, classe_esq, f"{nome_base}_LEFT.jpg")
        cv2.imwrite(caminho_salvar_esq, recorte_esq)
        
        # Salva Lado Direito
        classe_dir = 'com_ateroma' if row['lesao_direita'] else 'sem_ateroma'
        caminho_salvar_dir = os.path.join(pasta_destino, split_atual, classe_dir, f"{nome_base}_RIGHT.jpg")
        cv2.imwrite(caminho_salvar_dir, recorte_dir_espelhado)
        
        processados += 1

    print(f"Processamento concluído! {processados} pacientes divididos clinicamente.")

# --- Configurações de Execução ---
arquivo_labels = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Ateroma_1\Labels_Anotacoes_Consolidadas.csv" 
pasta_origem = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Ateroma_1"
pasta_destino = "Dataset_Recortado_Modelado"
processar_e_dividir_pacientes(pasta_origem, arquivo_labels, pasta_destino)