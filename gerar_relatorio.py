import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def gerar_relatorio(history, cm, classification_report, test_accuracy):
    # Criar diretório para o relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    relatorio_dir = f"relatorio_{timestamp}"
    os.makedirs(relatorio_dir, exist_ok=True)
    
    # 1. Gráfico de Acurácia
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(relatorio_dir, 'acuracia.png'))
    plt.close()
    
    # 2. Gráfico de Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss do Modelo por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(relatorio_dir, 'loss.png'))
    plt.close()
    
    # 3. Matriz de Confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.savefig(os.path.join(relatorio_dir, 'matriz_confusao.png'))
    plt.close()
    
    # 4. Relatório em Texto
    with open(os.path.join(relatorio_dir, 'relatorio.txt'), 'w') as f:
        f.write("=== Relatório de Treinamento ===\n\n")
        f.write(f"Data e Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        f.write("1. Métricas Finais:\n")
        f.write(f"Acurácia no conjunto de teste: {test_accuracy:.2%}\n\n")
        
        f.write("2. Resumo do Modelo:\n")
        f.write("Arquitetura CNN com:\n")
        f.write("- 3 camadas convolucionais (32, 64, 128 filtros)\n")
        f.write("- Camadas de MaxPooling\n")
        f.write("- Camada densa com 512 neurônios\n")
        f.write("- Dropout de 50%\n")
        f.write("- Saída binária (sigmoid)\n\n")
        
        f.write("3. Parâmetros de Treinamento:\n")
        f.write(f"- Tamanho da imagem: 150x150 pixels\n")
        f.write(f"- Batch size: 32\n")
        f.write(f"- Número de épocas: 20\n")
        f.write(f"- Otimizador: Adam (lr=0.0001)\n")
        f.write(f"- Função de perda: binary_crossentropy\n\n")
        
        f.write("4. Relatório de Classificação:\n")
        f.write(classification_report)
        
        f.write("\n5. Observações:\n")
        f.write("- O modelo foi treinado com data augmentation\n")
        f.write("- As imagens foram normalizadas (valores entre 0 e 1)\n")
        f.write("- O dataset foi dividido em 80% treino e 20% validação\n")
    
    print(f"Relatório gerado com sucesso em: {relatorio_dir}")

if __name__ == "__main__":
    # Exemplo de uso:
    # history = model.history
    # cm = confusion_matrix(y_true, y_pred)
    # classification_report = classification_report(y_true, y_pred)
    # test_accuracy = 0.90  # Substitua pelo valor real
    
    # gerar_relatorio(history, cm, classification_report, test_accuracy)
    print("Este script deve ser importado e usado com os resultados do treinamento.") 