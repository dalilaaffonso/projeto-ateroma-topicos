import os

def aplicar_prefixo_em_lote(diretorio, prefixo):
    if not os.path.exists(diretorio):
        print(f"Pasta não encontrada: {diretorio}")
        return

    itens = os.listdir(diretorio)
    contador = 0

    for nome in itens:
        # Evita renomear arquivos que já receberam o prefixo anteriormente
        if nome.startswith(prefixo):
            continue
        
        caminho_antigo = os.path.join(diretorio, nome)
        
        # Cria o novo nome garantindo que a formatação fique como PUB_1.png ou GV_1.png
        novo_nome = f"{prefixo}{nome}"
        caminho_novo = os.path.join(diretorio, novo_nome)
        
        os.rename(caminho_antigo, caminho_novo)
        contador += 1
        
    print(f"[{prefixo}] {contador} itens renomeados com sucesso na pasta.")

# 1. Configurações de Caminho
# Substitua pelos caminhos exatos onde estão as imagens originais de cada lote
pasta_base_publica = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Ateroma_1\Publica\dataset_semCAC"
pasta_base_gv = r"C:\Users\NOTEBOOK\Documents\Mestrado\Projeto_Ateroma_Tópicos\Dataset_Ateroma_1\GV\Dataset_semCAC_GV"

# 2. Execução da Renomeação
print("Iniciando renomeação em lote...")
#aplicar_prefixo_em_lote(pasta_base_publica, "PUB_")
aplicar_prefixo_em_lote(pasta_base_gv, "SCAC_")
print("Processo concluído! Suas pastas estão prontas para o CSV unificado.")