import os
import shutil
from sklearn.model_selection import train_test_split

def criar_validacao_externa():
    dir_origem = "Dataset_Recortado_Modelado"
    dir_destino = "Dataset_Externa"
    
    # 1. Criar estrutura de pastas
    for split in ['treino', 'validacao', 'teste']:
        for classe in ['com_ateroma', 'sem_ateroma']:
            os.makedirs(os.path.join(dir_destino, split, classe), exist_ok=True)
            
    todas_imagens = []
    
    # 2. Varrer todas as imagens já recortadas
    for root, _, files in os.walk(dir_origem):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                classe = 'com_ateroma' if 'com_ateroma' in root else 'sem_ateroma'
                todas_imagens.append((os.path.join(root, f), f, classe))
                
    # 3. Separar GV (Treino/Validação) e Base Pública (Teste)
    gv_com, gv_sem = [], []
    pub_com, pub_sem = [], []
    
    for caminho, nome, classe in todas_imagens:
        # Usa o identificador 'GV' no nome do arquivo (ajuste se necessário)
        if 'GV' in nome.upper():
            if classe == 'com_ateroma': gv_com.append((caminho, nome))
            else: gv_sem.append((caminho, nome))
        else:
            if classe == 'com_ateroma': pub_com.append((caminho, nome))
            else: pub_sem.append((caminho, nome))
            
    # 4. Divisão 80/20 do dataset GV para Treino e Validação interna
    gv_com_treino, gv_com_val = train_test_split(gv_com, test_size=0.2, random_state=42) if len(gv_com) > 0 else ([], [])
    gv_sem_treino, gv_sem_val = train_test_split(gv_sem, test_size=0.2, random_state=42) if len(gv_sem) > 0 else ([], [])
    
    def copiar(lista, split, classe):
        for caminho, nome in lista:
            shutil.copy(caminho, os.path.join(dir_destino, split, classe, nome))
            
    # GV vai para Treino e Validação
    copiar(gv_com_treino, 'treino', 'com_ateroma')
    copiar(gv_com_val, 'validacao', 'com_ateroma')
    copiar(gv_sem_treino, 'treino', 'sem_ateroma')
    copiar(gv_sem_val, 'validacao', 'sem_ateroma')
    
    # Base Pública vai EXCLUSIVAMENTE para Teste
    copiar(pub_com, 'teste', 'com_ateroma')
    copiar(pub_sem, 'teste', 'sem_ateroma')
    
    print(f"\n✅ Validação Externa Pronta em '{dir_destino}'!")
    print(f"Treino (Apenas GV): {len(gv_com_treino) + len(gv_sem_treino)} imagens")
    print(f"Validação (Apenas GV): {len(gv_com_val) + len(gv_sem_val)} imagens")
    print(f"Teste Inédito (Apenas Base Pública): {len(pub_com) + len(pub_sem)} imagens")

if __name__ == '__main__':
    criar_validacao_externa()