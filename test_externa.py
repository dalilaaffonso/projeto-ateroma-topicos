import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from models import CACClassifier
from dataset_augmentation import teste_transforms

def testar_validacao_externa():
    # ==========================================
    # 1. CONFIGURAÇÕES - TESTE EXTERNO
    # ==========================================
    diretorio_teste = "Dataset_Externa/teste" 
    ARQUITETURA_ATUAL = 'efficientnet_b4'
    caminho_pesos = f"melhor_modelo_externa_{ARQUITETURA_ATUAL}.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Teste de Validação Externa (Cross-Dataset) em: {device}")

    if not os.path.exists(caminho_pesos):
        print(f"Erro: Arquivo de pesos '{caminho_pesos}' não encontrado.")
        return

    # ==========================================
    # 2. CARREGAMENTO DOS DADOS (APENAS BASE PÚBLICA)
    # ==========================================
    dataset_teste = datasets.ImageFolder(diretorio_teste, transform=teste_transforms)
    dataloader_teste = DataLoader(dataset_teste, batch_size=8, shuffle=False)
    nomes_classes = dataset_teste.classes

    print(f"Avaliando o modelo (treinado em GV) nas {len(dataset_teste)} imagens da Base Pública...")

    # ==========================================
    # 3. INFERÊNCIA LIMPA (LIMIAR 0.5)
    # ==========================================
    modelo = CACClassifier(model_name=ARQUITETURA_ATUAL, num_classes=1, pretrained=False)
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=device))
    modelo = modelo.to(device)
    modelo.eval()
    
    y_verdadeiro = []
    y_predito = []

    with torch.no_grad():
        for inputs, labels in dataloader_teste:
            inputs = inputs.to(device)
            outputs = modelo(inputs)
            
            # Limiar padrão e sem otimizações (Baseline)
            probs = torch.sigmoid(outputs).squeeze()
            if probs.dim() == 0: probs = probs.unsqueeze(0)
            
            preds = (probs >= 0.50).cpu().numpy().astype(int)
            
            y_predito.extend(preds)
            y_verdadeiro.extend(labels.numpy())

    # ==========================================
    # 4. MÉTRICAS E MATRIZ DE CONFUSÃO
    # ==========================================
    cm = confusion_matrix(y_verdadeiro, y_predito)
    tn, fp, fn, tp = cm.ravel()
    
    acuracia = accuracy_score(y_verdadeiro, y_predito)
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print("\n" + "="*60)
    print("MÉTRICAS FINAIS - VALIDAÇÃO EXTERNA (BASELINE LIMPO)")
    print("="*60)
    print(f"Acurácia Global:   {acuracia:.4f}")
    print(f"Sensibilidade:     {sensibilidade:.4f} (Acerto em Doentes)")
    print(f"Especificidade:    {especificidade:.4f} (Acerto em Sadios)")
    print("\nRelatório Detalhado:")
    print(classification_report(y_verdadeiro, y_predito, target_names=nomes_classes))

    # ==========================================
    # 5. PLOTAGEM
    # ==========================================
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title(f'Validação Externa (Treino GV -> Teste Público)')
    plt.ylabel('Rótulo Verdadeiro (Base Pública)')
    plt.xlabel('Predição (Modelo GV)')
    
    nome_grafico = f'matriz_confusao_externa_{ARQUITETURA_ATUAL}.png'
    plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
    print(f"Gráfico '{nome_grafico}' salvo!")
    plt.show()

if __name__ == '__main__':
    testar_validacao_externa()