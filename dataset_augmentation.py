import torchvision.transforms as transforms

# 1. Pipeline de Treinamento (Com Aumento de Dados)
treino_transforms = transforms.Compose([
    # Redimensionamento padronizado para as redes convolucionais
    transforms.Resize((224, 224)),
    
    # Espelhamento horizontal para alinhar a anatomia direita/esquerda
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Rotação leve simulando inclinações do paciente no aparelho
    transforms.RandomRotation(degrees=15),
    
    # Variações de cor e contraste simulando diferentes equipamentos radiográficos
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    
    transforms.ToTensor(),
    
    # Normalização padrão da ImageNet (obrigatório para Transfer Learning)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Pipeline de Validação e Teste (Sem Aumento de Dados)
teste_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])