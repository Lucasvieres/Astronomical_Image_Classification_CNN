<div align="center">

# Classificador de Imagens Astronômicas usando CNN

| Disciplina | Semestre | Docente | Horário |
| :---: | :---: | :---: | :---: |
| PAM0466 - SISTEMAS INTELIGENTES | 2026.1 | PEDRO THIAGO VALÉRIO DE SOUZA | 3M23 4M45 |

| Discente                   | Matrícula  |
| :---:                      | :---:      |
| ELTON CAIO VIEIRA DE LIMA  | 2020010673 |
| LUCAS VIERES ARAÚJO FARIAS | 2025022531 |

</div>

<hr>

## INTRODUÇÃO

Este projeto implementa um modelo de Deep Learning focado na classificação de imagens astronômicas. Utilizando o framework PyTorch, foi construída uma Rede Neural Convolucional (CNN) para extrair características visuais e categorizar de forma autônoma imagens do espaço profundo (como galáxias, estrelas, nebulosas, etc.), utilizando o conjunto de dados `astro_dataset_maxia`.

## PRÉ-PROCESSAMENTO E ESTRUTURAÇÃO DOS DADOS

As imagens brutas passam por um pipeline rigoroso de transformações matemáticas utilizando a biblioteca `torchvision.transforms` antes de alimentarem a rede. O pipeline consiste em:

- **Conversão de Cores:** Garantia de que todas as imagens possuam 3 canais (RGB), evitando falhas com imagens em escala de cinza.
- **Redimensionamento:** Padronização das dimensões para matrizes de 128 x 128 pixels.
- **Tensorização:** Conversão das matrizes de imagem para tensores numéricos do PyTorch.
- **Normalização:** Ajuste da distribuição dos pixels (média 0.5 e desvio padrão 0.5), centralizando os valores no intervalo [-1, 1], o que acelera e estabiliza a convergência da rede durante o treinamento.

Os dados estão divididos em três conjuntos distintos utilizando o DataLoader (com lotes de 16 imagens): Treinamento (com embaralhamento de dados), Validação e Teste.

## ARQUITETURA DO MODELO (`AstroCNN`)

A arquitetura de rede neural convolucional desenvolvida (`AstroCNN`) foi dividida estruturalmente em dois grandes blocos:

### *Feature Extractor*

Composto por 4 blocos convolucionais em sequência. Cada bloco tem a função de mapear padrões cada vez mais complexos na imagem:

- **Camada Convolucional (`Conv2d`):** O número de filtros (mapas de características) dobra progressivamente em cada bloco: 32 -> 64 -> 128 -> 256, utilizando um kernel de tamanho 3x3 e padding de 1 para preservar as bordas.
- **Ativação Não-linear (`ReLU`):** Introduz a não-linearidade no modelo, zerando valores negativos.
- **Redução de Dimensionalidade (`MaxPool2d`):** Um filtro 2x2 atua reduzindo o tamanho espacial da imagem pela metade a cada bloco, diminuindo o custo computacional e destacando as feições mais relevantes.

### *Classifier*

A matriz de características final é achatada (*Flatten*) em um vetor unidimensional de base que alimenta uma rede Perceptron Multicamadas (MLP):

- **Camadas Densas:** Uma camada linear intermediária seguida por uma ativação ReLU, culminando em uma camada de saída correspondente ao número de classes do problema.
- **Regularização (`Dropout`):** Implementada com uma taxa de 0.3 (30% dos neurônios são desligados aleatoriamente durante o treino) para forçar a rede a aprender padrões genéricos, mitigando drasticamente o *overfitting*.

## PIPELINE DE TREINAMENTO

- **Função de Custo:** `CrossEntropyLoss`, padrão ouro para problemas de classificação multiclasse.
- **Otimizador:** `Adam`, com uma taxa de aprendizado de 10<sup>-3</sup>, garantindo uma descida de gradiente adaptativa.
- **Critério de Parada (*Early Stopping*):** O modelo é configurado para treinar por até 50 épocas. No entanto, conta com um mecanismo de parada antecipada com `paciência = 5`. Se o erro de validação (`val_loss`) não apresentar redução por 5 épocas consecutivas, o treinamento é interrompido.
- **Preservação de Pesos:** Durante o treinamento, o modelo rastreia a época em que obteve a menor taxa de erro na validação e salva esses pesos na memória.

## AVALIAÇÃO E MÉTRICAS

Ao fim do treinamento, os melhores pesos são restaurados. O modelo é então submetido ao conjunto de Testes (dados nunca vistos). O desempenho do modelo é mensurado qualitativamente e quantitativamente utilizando a biblioteca `scikit-learn`:

- **Acurácia Global (`Accuracy`)**
- ***Recall* Macro:** Avalia a capacidade do modelo de encontrar todas as amostras positivas de cada classe.
- ***F1-Score* Macro:** Média harmônica entre Precisão e Recall, sendo uma métrica robusta especialmente se o dataset for desbalanceado.
- **Matriz de Confusão:** Exibida no terminal para evidenciar as taxas de Falsos Positivos e Falsos Negativos entre classes específicas.