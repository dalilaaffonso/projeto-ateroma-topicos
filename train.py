import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Importa a sua classe classificadora e os pipelines criados
from models import CACClassifier
from dataset_augmentation import treino_transforms, teste_transforms

def carregar_dados(data_dir, batch_size=32):
    """Carrega as pastas estratificadas e aplica os transforms apropriados."""
    image_datasets = {
        'treino': datasets.ImageFolder(os.path.join(data_dir, 'treino'), transform=treino_transforms),
        'validacao': datasets.ImageFolder(os.path.join(data_dir, 'validacao'), transform=teste_transforms),
        'teste': datasets.ImageFolder(os.path.join(data_dir, 'teste'), transform=teste_transforms)
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'treino'), num_workers=2)
        for x in ['treino', 'validacao', 'teste']
    }
    
    return dataloaders, image_datasets

def treinar_modelo(modelo, dataloaders, criterio, otimizador, num_epocas=25, device='cpu'):
    """Loop de Otimização e Validação do Modelo."""
    melhor_acc = 0.0
    
    for epoca in range(num_epocas):
        print(f'\nÉpoca {epoca+1}/{num_epocas}')
        print('-' * 10)

        # Cada época tem uma fase de treino e uma de validação
        for fase in ['treino', 'validacao']:
            if fase == 'treino':
                modelo.train()
            else:
                modelo.eval()

            loss_acumulada = 0.0
            acertos = 0

            # Iteração sobre os lotes de dados
            for inputs, labels in tqdm(dataloaders[fase], desc=fase.capitalize()):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1) # Ajuste para BCELoss

                otimizador.zero_grad()

                with torch.set_grad_enabled(fase == 'treino'):
                    # Forward pass
                    outputs = modelo(inputs)
                    loss = criterio(outputs, labels)
                    
                    # Gera a decisão binária (limiar de 0.5)
                    preds = (outputs >= 0.5).float()

                    # Backward pass e otimização (somente no treino)
                    if fase == 'treino':
                        loss.backward()
                        otimizador.step()

                # Estatísticas
                loss_acumulada += loss.item() * inputs.size(0)
                acertos += torch.sum(preds == labels.data)

            # Cálculo final da métrica da época
            loss_epoca = loss_acumulada / len(dataloaders[fase].dataset)
            acc_epoca = acertos.double() / len(dataloaders[fase].dataset)

            print(f'{fase.capitalize()} Loss: {loss_epoca:.4f} Acc: {acc_epoca:.4f}')

            # Salva os pesos se o modelo apresentar melhor acurácia na validação
            if fase == 'validacao' and acc_epoca > melhor_acc:
                melhor_acc = acc_epoca
                torch.save(modelo.state_dict(), 'melhor_modelo_classificacao.pth')
                print(f"*** Novo melhor modelo salvo com Acc: {melhor_acc:.4f} ***")

    print(f'\nTreinamento Concluído! Melhor Acc de Validação: {melhor_acc:4f}')
    return modelo

if __name__ == '__main__':
    # 1. Configurações Iniciais
    diretorio_dados = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Dividido"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Treinando em: {device}")

    # 2. Carrega as imagens
    dataloaders, image_datasets = carregar_dados(diretorio_dados, batch_size=4)

    # 3. Instancia o Classificador (EfficientNet-B4 estabelecida no referencial)
    modelo = CACClassifier(model_name='efficientnet_b4', pretrained=True)
    modelo = modelo.to(device)

    # 4. Define Função de Custo e Otimizador
    # A BCELoss é a entropia cruzada para o nosso problema binário
    criterio = nn.BCELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=1e-4)

    # 5. Inicia o Treinamento
    modelo_treinado = treinar_modelo(modelo, dataloaders, criterio, otimizador, num_epocas=25, device=device)