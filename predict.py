import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random

def carregar_e_preparar_imagem(caminho_imagem, tamanho=(224, 224)):
    """Carrega e prepara a imagem para predição."""
    img = Image.open(caminho_imagem).convert('RGB')  # Converter para RGB
    img = img.resize(tamanho)
    img_array = np.array(img)
    
    # Normalizar
    img_array = img_array.astype('float32') / 255.0
    return img_array, img

def adicionar_resultado(imagem, predicao, probabilidade, classe_real):
    """Adiciona o resultado da predição na imagem."""
    # Converter para RGB se não estiver
    if imagem.mode != 'RGB':
        imagem = imagem.convert('RGB')
    
    draw = ImageDraw.Draw(imagem)
    
    # Definir cores fixas para cada classe
    COR_COVID = '#FF0000'  # Vermelho para COVID
    COR_NORMAL = '#00FF00'  # Verde para Normal
    
    # Definir cores baseado nas classes (real e predita)
    cor_real = COR_COVID if classe_real == "COVID" else COR_NORMAL
    cor_pred = COR_COVID if predicao == "COVID" else COR_NORMAL
    
    # Adicionar texto da classe real
    texto_real = f"Real: {classe_real}"
    posicao_real = (10, 10)
    
    # Adicionar texto da predição
    texto_pred = f"Pred: {predicao} ({probabilidade:.1%})"
    posicao_pred = (10, 30)
    
    # Desenhar texto da classe real
    draw.text((posicao_real[0]-1, posicao_real[1]), texto_real, fill='black')
    draw.text((posicao_real[0]+1, posicao_real[1]), texto_real, fill='black')
    draw.text((posicao_real[0], posicao_real[1]-1), texto_real, fill='black')
    draw.text((posicao_real[0], posicao_real[1]+1), texto_real, fill='black')
    draw.text(posicao_real, texto_real, fill=cor_real)
    
    # Desenhar texto da predição
    draw.text((posicao_pred[0]-1, posicao_pred[1]), texto_pred, fill='black')
    draw.text((posicao_pred[0]+1, posicao_pred[1]), texto_pred, fill='black')
    draw.text((posicao_pred[0], posicao_pred[1]-1), texto_pred, fill='black')
    draw.text((posicao_pred[0], posicao_pred[1]+1), texto_pred, fill='black')
    draw.text(posicao_pred, texto_pred, fill=cor_pred)
    
    # Adicionar indicador de acerto/erro
    acerto = predicao == classe_real
    texto_status = "✓" if acerto else "✗"
    posicao_status = (160, 20)
    cor_status = '#00FF00' if acerto else '#FF0000'  # Verde para acerto, Vermelho para erro
    
    draw.text((posicao_status[0]-1, posicao_status[1]), texto_status, fill='black')
    draw.text((posicao_status[0]+1, posicao_status[1]), texto_status, fill='black')
    draw.text((posicao_status[0], posicao_status[1]-1), texto_status, fill='black')
    draw.text((posicao_status[0], posicao_status[1]+1), texto_status, fill='black')
    draw.text(posicao_status, texto_status, fill=cor_status)
    
    return imagem

def fazer_predicao(modelo, imagem_path):
    """Faz a predição para uma imagem."""
    # Carregar e preparar a imagem
    img_array, img_original = carregar_e_preparar_imagem(imagem_path)
    
    # Expandir dimensões para batch
    img_batch = np.expand_dims(img_array, 0)
    
    # Fazer predição
    predicao = modelo.predict(img_batch, verbose=0)[0][0]
    
    # Determinar a classe real baseada no nome do arquivo
    classe_real = "Normal" if "normal" in imagem_path.lower() else "COVID"
    
    # Determinar a classe predita
    classe_pred = "Normal" if predicao > 0.5 else "COVID"
    prob = predicao if predicao > 0.5 else 1 - predicao
    
    # Adicionar resultado na imagem
    img_com_resultado = adicionar_resultado(img_original, classe_pred, prob, classe_real)
    
    return img_com_resultado, classe_pred, prob, classe_real

def main():
    # Carregar o modelo treinado
    print("Carregando o modelo...")
    modelo = tf.keras.models.load_model('modelo_final.h5')
    
    # Diretório com imagens de teste
    diretorio_teste = "dataset/test"
    
    # Criar diretório para salvar resultados
    os.makedirs("resultados_predicao", exist_ok=True)
    
    # Configurar número de imagens para cada classe
    n_imagens_por_classe = 15  # Aumentando para 15 imagens de cada classe
    
    # Selecionar algumas imagens de exemplo
    exemplos = {
        'covid': os.path.join(diretorio_teste, 'covid'),
        'normal': os.path.join(diretorio_teste, 'normal')
    }
    
    total_corretas = 0
    total_imagens = 0
    
    for classe, diretorio in exemplos.items():
        print(f"\nProcessando imagens da classe {classe}...")
        try:
            # Pegar todas as imagens do diretório
            todas_imagens = os.listdir(diretorio)
            # Embaralhar e selecionar n imagens
            imagens_selecionadas = random.sample(todas_imagens, n_imagens_por_classe)
            
            for img_nome in imagens_selecionadas:
                caminho_completo = os.path.join(diretorio, img_nome)
                print(f"\nAnalisando imagem: {img_nome}")
                
                # Fazer predição
                img_resultado, classe_pred, prob, classe_real = fazer_predicao(modelo, caminho_completo)
                
                # Salvar resultado
                nome_saida = f"resultados_predicao/predicao_{img_nome}"
                img_resultado.save(nome_saida)
                
                # Atualizar estatísticas
                total_imagens += 1
                if classe_pred == classe_real:
                    total_corretas += 1
                
                print(f"Classe real: {classe_real}")
                print(f"Predição: {classe_pred} (Probabilidade: {prob:.1%})")
                print(f"Resultado salvo em: {nome_saida}")
        
        except Exception as e:
            print(f"Erro ao processar diretório {classe}: {str(e)}")
    
    # Mostrar estatísticas finais
    if total_imagens > 0:
        acuracia = (total_corretas / total_imagens) * 100
        print(f"\nEstatísticas finais:")
        print(f"Total de imagens processadas: {total_imagens}")
        print(f"Predições corretas: {total_corretas}")
        print(f"Acurácia: {acuracia:.1f}%")

if __name__ == "__main__":
    main() 