# type: ignore
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import logging

# Configurações
IMG_SIZE = 224  # Tamanho padrão para DenseNet
BATCH_SIZE = 16
EPOCHS = 2
INITIAL_LR = 1e-4

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def criar_modelo():
    """Cria um modelo baseado em DenseNet121 com transfer learning."""
    # Carregar modelo base pré-treinado
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Congelar camadas do modelo base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Adicionar camadas personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def criar_data_generators():
    """Cria geradores de dados com augmentation apropriado para imagens médicas."""
    # Data augmentation para treino
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0,
        validation_split=0.2
    )
    
    return train_datagen

def carregar_dados(train_datagen):
    """Carrega os dados usando os geradores."""
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def criar_callbacks(nome_modelo='melhor_modelo.h5'):
    """Cria callbacks para o treinamento."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            nome_modelo,
            monitor='val_accuracy',
            save_best_weights_only=True,
            verbose=1
        )
    ]

def plotar_historico(history, nome_arquivo='historico_treinamento.png'):
    """Plota o histórico de treinamento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de acurácia
    ax1.plot(history.history['accuracy'], label='Treino')
    ax1.plot(history.history['val_accuracy'], label='Validação')
    ax1.set_title('Acurácia do Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico de loss
    ax2.plot(history.history['loss'], label='Treino')
    ax2.plot(history.history['val_loss'], label='Validação')
    ax2.set_title('Loss do Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()

def plotar_matriz_confusao(y_true, y_pred, nome_arquivo='matriz_confusao.png'):
    """Plota a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.savefig(nome_arquivo)
    plt.close()
    return cm

def plotar_curva_roc(y_true, y_pred_proba, nome_arquivo='curva_roc.png'):
    """Plota a curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(nome_arquivo)
    plt.close()
    return roc_auc

def predizer_imagem(model, caminho_imagem):
    """Faz a predição para uma única imagem."""
    img = load_img(caminho_imagem, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    probabilidade = prediction[0][0]
    classe = 'COVID' if probabilidade > 0.5 else 'Normal'
    
    return classe, probabilidade

def treinar_modelo():
    """Função principal de treinamento."""
    logger.info("Iniciando treinamento do modelo...")
    
    # Criar geradores de dados
    train_datagen = criar_data_generators()
    train_generator, val_generator = carregar_dados(train_datagen)
    
    logger.info(f"Classes encontradas: {train_generator.class_indices}")
    logger.info(f"Número de imagens de treino: {train_generator.samples}")
    logger.info(f"Número de imagens de validação: {val_generator.samples}")
    
    # Criar e compilar modelo
    model, base_model = criar_modelo()
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Primeira fase: treinar apenas as camadas superiores
    logger.info("Fase 1: Treinando camadas superiores...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=criar_callbacks('modelo_fase1.h5'),
        verbose=1
    )
    
    # Segunda fase: fine-tuning
    logger.info("Fase 2: Fine-tuning do modelo...")
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR/10),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=criar_callbacks('modelo_final.h5'),
        verbose=1
    )
    
    # Avaliação final
    logger.info("Avaliando modelo...")
    predictions = model.predict(val_generator)
    y_pred = (predictions > 0.5).astype(int)
    y_true = val_generator.classes
    
    # Gerar métricas e visualizações
    plotar_historico(history_fine)
    cm = plotar_matriz_confusao(y_true, y_pred)
    roc_auc = plotar_curva_roc(y_true, predictions)
    
    # Gerar relatório de classificação
    report = classification_report(y_true, y_pred)
    logger.info("\nRelatório de Classificação:")
    logger.info(f"\n{report}")
    
    # Salvar relatório em arquivo
    with open('relatorio_classificacao.txt', 'w') as f:
        f.write("=== Relatório de Classificação ===\n\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write("Matriz de Confusão:\n")
        f.write(str(cm))
        f.write("\n\nRelatório Detalhado:\n")
        f.write(report)
    
    # Testar algumas imagens de exemplo
    logger.info("\nTestando algumas imagens de exemplo...")
    for classe in ['covid', 'normal']:
        diretorio = os.path.join('dataset/train', classe)
        imagens = os.listdir(diretorio)[:3]
        
        for imagem in imagens:
            caminho_imagem = os.path.join(diretorio, imagem)
            classe_predita, prob = predizer_imagem(model, caminho_imagem)
            logger.info(f"\nImagem: {imagem}")
            logger.info(f"Classe real: {classe}")
            logger.info(f"Classe predita: {classe_predita}")
            logger.info(f"Probabilidade: {prob:.2%}")
    
    logger.info("\nTreinamento concluído com sucesso!")
    return model

if __name__ == "__main__":
    modelo = treinar_modelo()
