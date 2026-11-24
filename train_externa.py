import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import numpy as np

# Importa a sua classe classificadora e as transformações
from models import CACClassifier
from dataset_augmentation import treino_transforms, teste_transforms

def treinar_validacao_externa():
    # ==========================================
    # 1. CONFIGURAÇÕES - VALIDAÇÃO EXTERNA
    # ==========================================
    DIR_TREINO = "Dataset_Externa/treino"
    DIR_VAL = "Dataset_Externa/validacao"
    ARQUITETURA_ATUAL = 'efficientnet_b4' 

    EPOCHS = 25
    BATCH_SIZE = 4 
    LEARNING_RATE = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Validação Externa | Device: {device} | Arquitetura: {ARQUITETURA_ATUAL.upper()}")

    # ==========================================
    # 2. CARREGAMENTO DOS DADOS (APENAS GV)
    # ==========================================
    dataset_treino = datasets.ImageFolder(DIR_TREINO, transform=treino_transforms)
    dataset_val = datasets.ImageFolder(DIR_VAL, transform=teste_transforms)
    
    train_loader = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Imagens de Treino (GV): {len(dataset_treino)} | Validação (GV): {len(dataset_val)}")

    # ==========================================
    # 3. PREPARAÇÃO DO MODELO E OTIMIZAÇÃO (V0)
    # ==========================================
    model = CACClassifier(model_name=ARQUITETURA_ATUAL, num_classes=1, pretrained=True).to(device)
    
    # Loss padrão (sem pesos) e Limiar padrão (0.5) para estabelecer a base da Validação Externa
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0

    # ==========================================
    # 4. LOOP DE TREINO
    # ==========================================
    for epoch in range(EPOCHS):
        # Fase de Treinamento
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Treino]"):
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        # Fase de Validação
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)
                
        epoch_val_acc = correct_val / total_val
        print(f"Época {epoch+1} | Val Acc: {epoch_val_acc:.4f} | Val Loss: {val_loss/total_val:.4f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), f"melhor_modelo_externa_{ARQUITETURA_ATUAL}.pth")
            print(f"🏆 Novo melhor modelo salvo! (Acc: {best_val_acc:.4f})")

    print(f"\nTreinamento de GV Concluído! Melhor Acurácia de Validação: {best_val_acc:.4f}")

if __name__ == '__main__':
    treinar_validacao_externa()