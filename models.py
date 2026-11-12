import torch
import torch.nn as nn
import torchvision.models as models

class CACClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b4', num_classes=1, pretrained=True):
        super(CACClassifier, self).__init__()
        
        # Carregamento dinâmico da arquitetura escolhida
        if model_name == 'efficientnet_b4':
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b4(weights=weights)
            num_ftrs = self.backbone.classifier[1].in_features
            # Substitui a camada final para classificação binária
            self.backbone.classifier[1] = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid() 
            )
            
        elif model_name == 'resnet101':
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid()
            )
        
        # O mesmo padrão pode ser adicionado para DenseNet121, VGG16 e MobileNetV2.

    def forward(self, x):
        return self.backbone(x)