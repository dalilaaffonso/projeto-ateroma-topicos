import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Importa a arquitetura e as transformações limpas de teste
from models import CACClassifier
from dataset_augmentation import teste_transforms

def avaliar_modelo():
    # 1. Configurações e Caminhos
    diretorio_teste = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Recortado_Modelado\teste"
    caminho_pesos = "melhor_modelo_classificacao.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. Carregamento dos Dados
    dataset_teste = datasets.ImageFolder(diretorio_teste, transform=teste_transforms)
    dataloader_teste = DataLoader(dataset_teste, batch_size=8, shuffle=False, num_workers=2)
    
    # Nomes das classes gerados automaticamente (ex: 'com_ateroma', 'sem_ateroma')
    nomes_classes = dataset_teste.classes

    # 3. Preparação do Modelo
    # Pretrained=False porque vamos carregar os nossos próprios pesos treinados
    modelo = CACClassifier(model_name='efficientnet_b4', pretrained=False)
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=device))
    modelo = modelo.to(device)
    modelo.eval()

    y_verdadeiro = []
    y_predito = []

    print(f"Iniciando inferência em {len(dataset_teste)} imagens de teste...")

    # 4. Loop de Inferência
    with torch.no_grad():
        for inputs, labels in dataloader_teste:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = modelo(inputs)
            preds = (outputs >= 0.5).float().squeeze()
            
            # Ajuste para garantir que lidamos corretamente com lotes de tamanho 1
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
                
            y_verdadeiro.extend(labels.cpu().numpy())
            y_predito.extend(preds.cpu().numpy())

    # 5. Cálculo das Métricas Clínicas
    cm = confusion_matrix(y_verdadeiro, y_predito)
    tn, fp, fn, tp = cm.ravel()
    
    acuracia = accuracy_score(y_verdadeiro, y_predito)
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Capacidade de detectar a lesão
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Capacidade de liberar pacientes sadios
    
    print("\n" + "="*40)
    print("MÉTRICAS CLÍNICAS GERAIS")
    print("="*40)
    print(f"Acurácia Global:   {acuracia:.4f}")
    print(f"Sensibilidade:     {sensibilidade:.4f} (Recall da classe positiva)")
    print(f"Especificidade:    {especificidade:.4f} (Taxa de Verdadeiros Negativos)")
    print("\nRelatório Detalhado (Precision, F1-Score):")
    print(classification_report(y_verdadeiro, y_predito, target_names=nomes_classes))

    # 6. Plotagem da Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title('Matriz de Confusão - Teste Baseline')
    plt.ylabel('Rótulo Verdadeiro (Ground Truth)')
    plt.xlabel('Predição da EfficientNet-B4')
    
    # Salva o gráfico na sua pasta para usar nos slides
    plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
    print("Gráfico 'matriz_confusao.png' salvo com sucesso na pasta do projeto.")
    plt.show()

if __name__ == '__main__':
    avaliar_modelo()