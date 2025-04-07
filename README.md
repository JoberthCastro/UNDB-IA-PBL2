# Detecção de COVID-19 em Radiografias com Redes Neurais Convolucionais

Este projeto tem como objetivo desenvolver um modelo de inteligência artificial capaz de classificar radiografias de tórax como **COVID-19** ou **Normal**, utilizando redes neurais convolucionais (CNNs). A iniciativa visa auxiliar no diagnóstico rápido da doença, especialmente em ambientes clínicos com escassez de profissionais especializados.

## 👨‍🏫 Orientador
Prof. Dr. Giovanni Lucca

## 👥 Equipe
- Anderson Aires 
- Joberth Castro 
- Maria Clara
- Wesley Gomes 
  
---

## 📌 Motivação

Durante a pandemia da COVID-19, ficou evidente a necessidade de métodos diagnósticos rápidos e acessíveis. Radiografias de tórax são uma alternativa viável ao RT-PCR, principalmente em regiões com infraestrutura limitada. A IA se mostra uma ferramenta poderosa para ampliar a capacidade de diagnóstico.

---

## 🎯 Objetivo

Desenvolver um modelo preditivo baseado em CNNs que classifique imagens de raio-x do tórax em duas categorias:
- ✅ Normal
- ❌ COVID-19

---

## 🧠 Metodologia

### 🔸 Base de Dados
- [COVID-19 Radiography Database - Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data)

### 🔸 Etapas do Processamento
1. **Redimensionamento**: 224x224 pixels  
2. **Normalização**: Valores de pixel entre 0 e 1  
3. **Aumento de Dados**: Rotação, espelhamento e zoom  

### 🔸 Arquitetura do Modelo (CNN)
- 3 Camadas Convolucionais (32, 64, 128 filtros)
- Camadas de MaxPooling
- Camada Densa (512 neurônios) com Dropout (0.5)
- Função de ativação: ReLU / Softmax
- Divisão dos dados:  
  - 70% Treino  
  - 15% Validação  
  - 15% Teste  

---

## 📊 Resultados

- O modelo apresentou excelente desempenho mesmo com um dataset moderado.
- As CNNs se mostraram eficazes para a detecção automática de COVID-19 em imagens radiográficas.

---

## 🔮 Trabalhos Futuros

- Aumentar a base de dados para melhorar a capacidade de generalização do modelo.
- Expandir o escopo do modelo para incluir detecção de outras doenças pulmonares, como pneumonia e tuberculose.

---

## 📝 Conclusão

Este projeto demonstra o potencial da inteligência artificial na área médica, oferecendo soluções práticas e rápidas para o diagnóstico de doenças respiratórias em ambientes com recursos limitados.

---

## 📌 Licença
Este projeto é apenas para fins educacionais.

