# VISHOD - Visual Dataset Scraping and Hybrid Outlier Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Sobre o Projeto

**VISHOD** (Visual Dataset Scraping and Hybrid Outlier Detection) é um método automatizado para construção e refinamento de datasets de imagens utilizando web scraping e uma abordagem híbrida de detecção de outliers. Este projeto foi desenvolvido como parte de uma pesquisa encabeçada pelo Prof. Seruffo e o Orientando FLávio Moura, focando na criação de datasets de alta qualidade para aplicações de aprendizado supervisionado, especialmente em contextos com recursos computacionais limitados.

### 🎯 Objetivo

O projeto visa resolver o desafio de construir datasets extensos e precisamente rotulados de forma automatizada, sem necessidade de anotação manual. Como estudo de caso, desenvolvemos um classificador leve de imagens para reconhecimento de formas geométricas bidimensionais comuns, destinado a apoiar aplicações educacionais em contextos com conectividade limitada, como a região amazônica.

### 📊 Resultados Principais

Os resultados demonstram que o método permite a construção de datasets balanceados e de alta qualidade, com melhorias significativas após o processo de limpeza:

- **+25.6%** de aumento na variância média do PCA (maior diversidade)
- **-5.6%** de redução na distância média ao centróide (maior coesão)
- **+21.6%** de aumento no índice de similaridade estrutural médio (SSIM) (maior homogeneidade visual)

## 🏗️ Arquitetura do Sistema

O pipeline proposto compreende quatro estágios principais:

### 1. Coleta Automatizada de Imagens (Web Scraping)
- Extração sistemática de imagens de fontes online
- Uso de termos combinados (descritores visuais e modificadores contextuais) para consultas em mecanismos de busca
- Paralelização do processo de coleta para eficiência

### 2. Extração de Representações Semânticas
- Utilização de modelos de deep learning pré-treinados (ResNet-50) para extração de features
- Geração de representações vetoriais de alta dimensão para análise posterior

### 3. Detecção e Eliminação de Outliers (Abordagem Híbrida)
- **HDBSCAN**: Detecção de outliers baseada em densidade hierárquica
- **Isolation Forest**: Identificação de anomalias através de isolamento
- **Distância de Mahalanobis**: Detecção de instâncias estatisticamente discrepantes
- Abordagem de consenso para identificação robusta de outliers

### 4. Validação do Dataset
- Inspeção visual automatizada
- Análise estatística de performance
- Métricas de qualidade: PCA, Distância Euclidiana e SSIM

## 📁 Estrutura do Projeto

```
WildShapesDB/
├── collector/          # Módulo de coleta de imagens via web scraping
│   ├── scrapper.py    # Implementação do scraper
│   └── utils.py       # Utilitários para geração de queries
├── cleaner/           # Módulo de detecção e remoção de outliers
│   ├── feature_extractor.py    # Extração de features usando deep learning
│   ├── hdbscan.py              # Detecção de outliers com HDBSCAN
│   ├── isolation_forest.py    # Detecção de outliers com Isolation Forest
│   └── hnsw_index.py          # Indexação eficiente com FAISS HNSW
├── evaluator/         # Módulo de avaliação de qualidade do dataset
│   ├── analyze.py     # Análise estatística do dataset
│   ├── metrics.py     # Implementação de métricas (PCA, SSIM, etc.)
│   └── healthcheck.py # Verificação de integridade das imagens
├── classifier/        # Módulo de classificação
│   ├── model.py       # Arquitetura do modelo (EfficientNet-B0 + Feature Fusion)
│   ├── training.py    # Script de treinamento
│   ├── evaluate.py    # Script de avaliação
│   ├── optimize.py    # Otimização de hiperparâmetros com Optuna
│   └── data.py        # Preparação e divisão dos dados
├── gen_dataset.py     # Script principal para geração do dataset
├── clean_dataset.py   # Script principal para limpeza do dataset
├── train_model.py     # Script principal para treinamento do modelo
├── optimize_model.py  # Script principal para otimização de hiperparâmetros
├── test.py            # Script de teste do modelo
└── search_query_data.json  # Dados de queries de busca por classe
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip ou conda

### Instalação das Dependências

```bash
pip install -r requirements.txt
```

### Principais Dependências

- **PyTorch 2.6.0**: Framework de deep learning
- **torchvision 0.21.0**: Modelos pré-treinados e transformações
- **scikit-learn 1.6.1**: Algoritmos de machine learning
- **hdbscan 0.8.40**: Clustering hierárquico baseado em densidade
- **faiss-cpu 1.10.0**: Biblioteca de busca vetorial eficiente
- **selenium 4.28.1**: Automação de navegador para web scraping
- **opencv-python 4.11.0.86**: Processamento de imagens
- **matplotlib 3.10.0**: Visualização de dados
- **pandas 2.2.3**: Manipulação de dados
- **optuna**: Otimização de hiperparâmetros

## 💻 Uso

### 1. Geração do Dataset

Execute o script para coletar imagens automaticamente:

```bash
python gen_dataset.py
```

Este script:
- Carrega as queries de busca do arquivo `search_query_data.json`
- Gera consultas combinadas usando adjetivos, classes e substantivos
- Executa o web scraping em paralelo para coletar imagens
- Organiza as imagens por classe geométrica

### 2. Limpeza do Dataset

Execute o script para detectar e remover outliers:

```bash
python clean_dataset.py
```

Este script:
- Extrai features das imagens usando ResNet-50
- Aplica três métodos de detecção de outliers (HDBSCAN, Isolation Forest, Mahalanobis)
- Remove outliers identificados por consenso
- Gera visualizações e análises estatísticas
- Avalia a qualidade do dataset antes e depois da limpeza

### 3. Treinamento do Modelo

Treine o classificador de formas geométricas:

```bash
python train_model.py
```

O modelo utiliza:
- **EfficientNet-B0** como backbone (pré-treinado no ImageNet)
- **Feature Fusion** combinando features do backbone com features de bordas
- **ECA (Efficient Channel Attention)** para atenção nas features
- **Early stopping** e **checkpointing** para melhor modelo

### 4. Otimização de Hiperparâmetros

Otimize os hiperparâmetros do modelo usando Optuna:

```bash
python optimize_model.py
```

Este script:
- Executa 100 trials de otimização
- Explora espaço de hiperparâmetros (learning rate, batch size, dropout, etc.)
- Salva os melhores hiperparâmetros encontrados
- Gera visualizações do processo de otimização

### 5. Avaliação do Modelo

Teste o modelo treinado:

```bash
python test.py
```

## 🧪 Modelo de Classificação

### Arquitetura

O classificador utiliza uma arquitetura híbrida:

1. **Backbone**: EfficientNet-B0 (pré-treinado, camadas iniciais congeladas)
2. **Extrator de Features de Bordas**: CNN leve para capturar características geométricas
3. **Feature Fusion**: Combinação de features do backbone e de bordas usando ECA
4. **Classificador**: MLP com BatchNorm e Dropout

### Classes Suportadas

O modelo classifica as seguintes **9 formas geométricas 2D**:
- Círculo (Circle)
- Elipse (Ellipse)
- Hexágono (Hexagon)
- Paralelogramo (Parallelogram)
- Pentágono (Pentagon)
- Retângulo (Rectangle)
- Quadrado (Square)
- Trapézio (Trapezoid)
- Triângulo (Triangle)

## 📈 Métricas de Qualidade do Dataset

O sistema avalia a qualidade do dataset usando três métricas principais:

1. **Variância do PCA**: Mede a diversidade dos dados no espaço de features
2. **Distância Euclidiana ao Centróide**: Mede a coesão e consistência dos dados
3. **SSIM (Structural Similarity Index)**: Mede a similaridade estrutural e homogeneidade visual

## 📊 Visualizações

O projeto gera automaticamente visualizações em `plots/`:

- Análises de clusters (HDBSCAN)
- Distribuições de anomalias (Isolation Forest)
- Distribuições de distância de Mahalanobis
- Análises comparativas do dataset antes e depois da limpeza

## 🎓 Aplicação Educacional

Este projeto foi desenvolvido para suportar a aplicação **GeoMeta**, um aplicativo móvel que auxilia no ensino de geometria através da classificação de formas 2D. O modelo foi otimizado para:

- Operação offline
- Compatibilidade com dispositivos de baixo desempenho
- Baixo consumo de recursos computacionais
- Alta precisão em reconhecimento de formas geométricas

## 📝 Configuração

### Arquivo de Queries de Busca

O arquivo `search_query_data.json` contém as configurações de busca para cada classe geométrica, incluindo:
- Objetos do cotidiano que representam cada forma
- Adjetivos descritivos
- Substantivos contextuais

### Divisão do Dataset

Os splits do dataset são salvos em `dataset_splits/`:
- `train_indices.pt`: Índices de treinamento
- `val_indices.pt`: Índices de validação
- `test_indices.pt`: Índices de teste

## 🔬 Metodologia de Detecção de Outliers

### Abordagem Híbrida de Consenso

O sistema utiliza três métodos complementares:

1. **HDBSCAN**: Identifica pontos que não pertencem a clusters densos
2. **Isolation Forest**: Isola anomalias através de árvores de decisão
3. **Distância de Mahalanobis**: Detecta pontos estatisticamente distantes da distribuição normal

Um ponto é considerado outlier se identificado por pelo menos dois dos três métodos (consenso).

## 📚 Referências

Este projeto é baseado na pesquisa:

**VISHOD - Visual Dataset Scraping and Hybrid Outlier Detection**

Autores: Flávio Moura, Vitor Melo, André Alves, Lyanh Pinto, Walter Júnior, Adriano Santos, Roberto Oliveira, Jefferson Morais, Diego Cardoso, e Marcos Seruffo

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 👥 Autores

- **Flávio Moura**
- **Vitor Melo**
- **André Alves**
- **Lyanh Pinto**
- **Walter Júnior**
- **Adriano Santos**
- **Roberto Oliveira**
- **Jefferson Morais**
- **Diego Cardoso**
- **Marcos Seruffo**

## 🙏 Agradecimentos

Agradecemos a todos os colaboradores e à comunidade de código aberto pelas ferramentas e bibliotecas que tornaram este projeto possível.
