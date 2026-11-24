import torchvision.transforms as transforms
# este é o módulo de processamento de imagens do PyTorch, com as funções matemáticas necessárias para manipular as matrizes 
# de pixels antes que elas sejam injetadas na rede neural para o treinamento

# 1. Pipeline de Treinamento (Com Aumento de Dados, mantendo a anatomia alinhada)
# o Compose é como uma linha de montagem sequencial, empacotando múltiplas transformações, garantindo que a imagem bruta entre no pipeline
# e passe por cada modificação matemática na ordem exata lista, saindo um tensor formatado para a IA.
treino_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), #esta operação converte a escala origianl de pixels (0 a 255) para valores flutuantes normalizados (0 a 1) e rearranja a estrutura 
    #da matriz para o padrão que a placa de video exige para realizar os cálculos das convoluções de forma eficiente
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #aplica a normalização estatística z-score utilizando a média e o desvio padrão das imagens do ImageNet, que é a base de dados que a rede neural 
    # foi pré-treinada, garantindo que os valores de entrada estejam na mesma escala que os dados originais. mean é a média e std é o desvio padrão
    # mean tem a média exata do brilho dos pixels para os canais de cor vermelho, verde e azul (RGB) e o std é o desvio padrão, ou seja, o quanto as cores variam em torno dessa média em cada um dos três canais
    # a normalização é uma etapa obrigatória para que o transfer learning funcione. Assim o Pytorch força os tons de cinza das radiografias a entrarem
    # na exata mesma escala numérica que os pesos da rede neural já conhecem e estão acostumadas a processar
])

# 2. Pipeline de Validação e Teste (Sem Aumento de Dados)
teste_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])