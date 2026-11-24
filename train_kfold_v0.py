import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from tqdm import tqdm

from models import CACClassifier
from dataset_augmentation import treino_transforms, teste_transforms

# ==========================================
# 1. CONFIGURAÇÕES - VERSÃO 0 (BASELINE LIMPO)
# ==========================================
DATA_DIR = "Dataset_Recortado_Modelado"
ARQUITETURA_ATUAL = 'efficientnet_b4' 

NUM_FOLDS = 5
EPOCHS = 25
BATCH_SIZE = 4 # Mantido baixo para não travar a memória
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device} | Arquitetura: {ARQUITETURA_ATUAL.upper()} | MODO: V0 (Sem Otimização)")

# ==========================================
# 2. CARREGAMENTO E AGRUPAMENTO (V0)
# ==========================================
dataset_completo = ImageFolder(root=DATA_DIR)
labels = dataset_completo.targets
pacientes = []

for caminho, _ in dataset_completo.samples:
    nome_arquivo = os.path.basename(caminho)
    partes_nome = nome_arquivo.split("_")
    id_paciente = f"{partes_nome[0]}_{partes_nome[1]}" 
    pacientes.append(id_paciente)

sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_results = []

import gc

# ==========================================
# 3. LOOP DE TREINO V0 (SEM PESOS NA LOSS)
# ==========================================
for fold, (train_idx, val_idx) in enumerate(sgkf.split(dataset_completo, labels, groups=pacientes)):
    print(f"\n{'='*40}\n🚀 INICIANDO V0 - FOLD {fold + 1}/{NUM_FOLDS}\n{'='*40}")
    
    train_subset = Subset(dataset_completo, train_idx)
    val_subset = Subset(dataset_completo, val_idx)
    train_subset.dataset.transform = treino_transforms
    val_subset.dataset.transform = teste_transforms
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = CACClassifier(model_name=ARQUITETURA_ATUAL, num_classes=1, pretrained=True).to(device)
    
    # REMOVIDO: pos_weight. Agora é uma Loss padrão (V0)!
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Fold {fold+1} - Treino Ep {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float() # Limiar padrão 50%
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float() # Limiar padrão 50%
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)
                
        epoch_val_acc = correct_val / total_val
        print(f"Fold {fold+1} | Época {epoch+1}/{EPOCHS} | Val Acc: {epoch_val_acc:.4f} | Val Loss: {val_loss/total_val:.4f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Salvando os pesos com o nome V0
            torch.save(model.state_dict(), f"melhor_modelo_v0_{ARQUITETURA_ATUAL}_fold_{fold+1}.pth")
            
    fold_results.append(best_val_acc)
    
    del model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nRESULTADO FINAL V0: Acurácia Média: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")