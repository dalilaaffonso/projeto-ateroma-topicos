import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, fbeta_score

# Importa a arquitetura e transformações limpas (sem augmentation no teste)
from models import CACClassifier
from dataset_augmentation import teste_transforms

def avaliar_ensemble_otimizado():
    # ==========================================
    # 1. CONFIGURAÇÕES
    # ==========================================
    # Ajuste o caminho se sua pasta de teste estiver em outro local
    diretorio_teste = "Dataset_Recortado_Modelado/teste" 
    ARQUITETURA_ATUAL = 'efficientnet_b4'
    NUM_FOLDS = 5
    
    # === Teste com 0.50 ===
    # LIMIAR_DECISAO = 0.50 

    # === Teste com 0.40 ===
    LIMIAR_DECISAO = 0.40 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando avaliação em: {device}")

    # ==========================================
    # 2. CARREGAMENTO DOS DADOS INÉDITOS (TESTE)
    # ==========================================
    dataset_teste = datasets.ImageFolder(diretorio_teste, transform=teste_transforms)
    dataloader_teste = DataLoader(dataset_teste, batch_size=8, shuffle=False)
    nomes_classes = dataset_teste.classes

    print(f"Realizando inferência Ensemble em {len(dataset_teste)} recortes de teste...")

    y_verdadeiro = []
    todas_probabilidades = [] # Armazenará as probabilidades de cada um dos 5 folds

    # ==========================================
    # 3. INFERÊNCIA COM OS 5 MODELOS (FOLDS DA V0)
    # ==========================================
    for fold in range(1, NUM_FOLDS + 1):
        # Carrega os pesos da V0 para testar o novo limiar na mesma rede!
        caminho_pesos = f"melhor_modelo_v0_{ARQUITETURA_ATUAL}_fold_{fold}.pth"
        
        if not os.path.exists(caminho_pesos):
            print(f"⚠️ Aviso: Pesos do fold {fold} não encontrados ({caminho_pesos}).")
            continue
            
        # Instancia o modelo para o fold atual
        modelo = CACClassifier(model_name=ARQUITETURA_ATUAL, num_classes=1, pretrained=False)
        modelo.load_state_dict(torch.load(caminho_pesos, map_location=device))
        modelo = modelo.to(device)
        modelo.eval()
        
        probabilidades_fold = []
        
        with torch.no_grad():
            for inputs, labels in dataloader_teste:
                inputs = inputs.to(device)
                outputs = modelo(inputs)
                
                # Aplica a função Sigmoid para obter a probabilidade exata (0 a 1)
                probs = torch.sigmoid(outputs).squeeze()
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)
                    
                probabilidades_fold.extend(probs.cpu().numpy())
                
                # Salva o gabarito real (y_verdadeiro) apenas no loop do primeiro fold
                if fold == 1: 
                    y_verdadeiro.extend(labels.numpy())
                    
        todas_probabilidades.append(probabilidades_fold)
        print(f"✅ Fold {fold} processado com sucesso.")

    # ==========================================
    # 4. ENSEMBLE VOTING E MÉTRICAS CONSOLIDADAS
    # ==========================================
    # Calcula a média da probabilidade dos 5 modelos para cada radiografia
    probabilidades_medias = np.mean(todas_probabilidades, axis=0)
    
    # === APLICANDO O NOVO LIMIAR ===
    y_predito = (probabilidades_medias >= LIMIAR_DECISAO).astype(int)

    cm = confusion_matrix(y_verdadeiro, y_predito)
    tn, fp, fn, tp = cm.ravel()
    
    acuracia = accuracy_score(y_verdadeiro, y_predito)
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # === CÁLCULO DO F2-SCORE (PRIORIZA DETECTAR A DOENÇA) ===
    f2_score_val = fbeta_score(y_verdadeiro, y_predito, beta=2)
    
    print("\n" + "="*60)
    print(f"MÉTRICAS FINAIS - LIMIAR AJUSTADO PARA {LIMIAR_DECISAO*100}%")
    print("="*60)
    print(f"Acurácia Global:   {acuracia:.4f}")
    print(f"Sensibilidade:     {sensibilidade:.4f} (Menos Falsos Negativos)")
    print(f"Especificidade:    {especificidade:.4f} (Capacidade de acertar sadios)")
    print(f"F2-Score Clínico:  {f2_score_val:.4f} <-- (Substitui o F1)")
    print("\nRelatório Detalhado (Precision, F1-Score):")
    print(classification_report(y_verdadeiro, y_predito, target_names=nomes_classes))

    # ==========================================
    # 5. PLOTAGEM DA MATRIZ DE CONFUSÃO GLOBAL
    # ==========================================
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title(f'Matriz de Confusão V1 (Limiar 50%) - {ARQUITETURA_ATUAL}')
    plt.ylabel('Rótulo Verdadeiro (Ground Truth)')
    plt.xlabel('Predição (Média dos 5 Folds)')
    
    nome_grafico = f'matriz_confusao_{ARQUITETURA_ATUAL}_v1_kfold.png'
    plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
    print(f"Gráfico '{nome_grafico}' salvo na pasta do projeto.")
    plt.show()

if __name__ == '__main__':
    avaliar_ensemble_otimizado()