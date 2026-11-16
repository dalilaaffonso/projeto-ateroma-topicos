import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset_augmentation import teste_transforms

from models import CACClassifier

def visualizar_gradcam():
    # 1. Configurações e Caminhos
    caminho_pesos = "melhor_modelo_classificacao.pth"
    # Pegando a primeira imagem de teste COM ATEROMA para provar a detecção
    pasta_ateroma = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Recortado_Modelado\teste\com_ateroma"
    
    # Lista os arquivos e pega o primeiro
    imagens = [f for f in os.listdir(pasta_ateroma) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imagens:
        print("Nenhuma imagem encontrada na pasta de teste.")
        return
    
    caminho_imagem = os.path.join(pasta_ateroma, imagens[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. Carrega o Modelo Treinado
    modelo = CACClassifier(model_name='efficientnet_b4', pretrained=False)
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=device))
    modelo = modelo.to(device)
    modelo.eval()

    # Varredura automática para encontrar a última camada convolucional
    def obter_ultima_camada_conv(mod):
        ultima_conv = None
        for camada in mod.modules():
            if isinstance(camada, torch.nn.Conv2d):
                ultima_conv = camada
        return ultima_conv

    target_layers = [obter_ultima_camada_conv(modelo)]

    # 3. Prepara a Imagem
    img_pil = Image.open(caminho_imagem).convert('RGB')
    input_tensor = teste_transforms(img_pil).unsqueeze(0).to(device)

    # Imagem base para sobreposição
    img_redimensionada = img_pil.resize((224, 224))
    img_base_rgb = np.float32(img_redimensionada) / 255.0

    # 4. Configura e Executa o Grad-CAM
    cam = GradCAM(model=modelo, target_layers=target_layers)
    
    # O target=0 (ou None) instrui a focar na classe de maior probabilidade predita
    targets = [ClassifierOutputTarget(0)] 
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 5. Mescla o Mapa de Calor com a Radiografia
    visualizacao = show_cam_on_image(img_base_rgb, grayscale_cam, use_rgb=True)

    # 6. Plotagem e Salvamento
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_redimensionada)
    plt.title('Radiografia Original (Corte)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualizacao)
    plt.title('Grad-CAM (Foco da IA)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('heatmap_gradcam.png', dpi=300)
    print("Mapa de Calor gerado e salvo como 'heatmap_gradcam.png'!")
    plt.show()

if __name__ == '__main__':
    visualizar_gradcam()