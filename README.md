# Projeto: Classificação Zero-Shot vs. Fine-Tuning para Análise de Sentimento

Este repositório contém o projeto final da disciplina **"Modelos de Linguagem e Generativos"**, focado na comparação de duas técnicas centrais de PLN: Classificação Zero-Shot e Fine-Tuning.

### Programa de pós-graduação em Computação Aplicada - Metrado Profissional - Mackenzie
* `Prof. Rogério de Oliveira`
### Autores
* `Gildo Manzi da Silva` - RA: 10329658
* `Rafael da Silva Rosa` - RA: 10746329
* `Rogério Goussain Labat` - RA: 10746326

---

## 🎯 Introdução

O objetivo deste projeto é comparar o desempenho de duas abordagens distintas para a tarefa de análise de sentimento (classificar reviews de filmes como "positivos" ou "negativos") usando um dataset com reviews do IMDB disponibilizado no Kaggle.

As duas abordagens comparadas são:

1.  **Classificação Zero-Shot (Baseline):** Utiliza um modelo generalista (`facebook/bart-large-mnli`) que classifica o texto sem nunca ter sido treinado especificamente nesta tarefa.
2.  **Fine-Tuning (Desafiante):** Utiliza um modelo especialista (`distilbert-base-uncased`) que é treinado (ajustado) em milhares de exemplos específicos da tarefa.

A hipótese inicial é que o modelo Fine-Tuned, por ser especialista, superaria o modelo Zero-Shot generalista.

---

## 🏆 Resultados Principais

Contrariando a hipótese inicial, o modelo Zero-Shot (Baseline) apresentou um desempenho similar ao modelo especialista Fine-Tuned.

| Estratégia | Modelo Base | Acurácia | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- |
| **Baseline (Zero-Shot)** | `facebook/bart-large-mnli` | **90%** | **0.90** |
| **Desafiante (Fine-Tuned)**| `distilbert-base-uncased`| **90%** | **0.90** |

**Análise da Conclusão:**
Acreditamos que para uma tarefa binária simples como esta, o poder generalista de um modelo de fundação de grande escala já é suficiente para atingir uma performance muito alta, alcançando o mesmo resultado de um modelo treinado para ser especializado na tarefa.
1.  O modelo **Zero-Shot (BART-Large)** possui 406M de parâmetros e foi treinado em uma tarefa (NLI) que se traduz muito bem para a análise de sentimento.
2.  O modelo **Fine-Tuned (DistilBERT)** é significativamente menor (66M de parâmetros).

---

## 🧠 Referencial Teórico

### Classificação Zero-Shot
A classificação Zero-Shot é uma técnica onde um modelo pode classificar dados em categorias que não viu durante o treinamento. No contexto de PLN, isso é comumente alcançado "reformulando" a tarefa de classificação como uma tarefa de **Inferência de Linguagem Natural (NLI)**. O modelo avalia a probabilidade de uma "premissa" (o review do filme) implicar logicamente uma "hipótese" (ex: "Este texto é positivo"). O modelo `bart-large-mnli` é pré-treinado na tarefa Multi-Genre NLI (MNLI), tornando-o ideal para isso.

### Fine-Tuning
Fine-Tuning (Ajuste Fino) é o processo de pegar um modelo de fundação pré-treinado (como o `DistilBERT`, que foi treinado para "entender" a linguagem em geral) e treiná-lo um pouco mais em um conjunto de dados específico e com uma tarefa específica. Uma "cabeça de classificação" é adicionada ao topo do modelo, e seus pesos são ajustados para se especializarem na nova tarefa (neste caso, classificar reviews do IMDB).

---

## 🧪 Metodologia

O projeto foi conduzido em duas etapas principais, ambas utilizando o dataset [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) baixado via `kagglehub`.

### 1. Baseline: Modelo Zero-Shot
* **Modelo:** `facebook/bart-large-mnli` (406M de parâmetros).
* **Processo:** 300 reviews do dataset foram selecionados aleatoriamente.
* **Execução:** O modelo `pipeline` foi carregado na GPU (`cuda:0`) e classificou as 300 amostras com os rótulos `['positive', 'negative']`.
* **Métricas:** Um `classification_report` (Acurácia, Precisão, Recall, F1-Score) foi gerado comparando as previsões com os rótulos reais.

### 2. Desafiante: Modelo Fine-Tuned
* **Modelo:** `distilbert-base-uncased` (66M de parâmetros).
* **Processo:**
    1.  Uma amostra de 4000 reviews foi selecionada do dataset.
    2.  Os dados foram divididos em 80% para treino (3200 amostras) e 20% para teste (800 amostras).
    3.  O `AutoTokenizer` foi usado para preparar os dados.
    4.  O modelo foi treinado por 3 épocas usando o `Trainer` da Hugging Face, com avaliação ao final de cada época.
    5.  Foi configurado para salvar apenas o melhor modelo (`load_best_model_at_end=True`), que se revelou o da Época 2, evitando overfitting que ocorreu na Época 3.
* **Métricas:** As mesmas métricas foram calculadas no conjunto de teste de 800 amostras.

---

## 🚀 Instruções de Uso

### Pré-requisitos
* Python 3.10+
* Uma GPU NVIDIA com CUDA (essencial para performance). O projeto foi testado com CUDA 12.9.
* Git

### 1. Clonar o Repositório
```bash
git clone [https://github.com/rafael-rosa/mack-modelos-linguagem-generativos.git](https://github.com/rafael-rosa/mack-modelos-linguagem-generativos.git)
cd mack-modelos-linguagem-generativos
```

### 2. Configurar o Ambiente Virtual

```bash
python -m venv .venv

# No Windows
.venv\Scripts\activate

# No Linux/macOS
source .venv/bin/activate
```

### 3. Instalar as Dependências

O projeto usa dois arquivos de requisitos. O other-requirements.txt força a instalação do PyTorch com o suporte a CUDA correto.

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r other-requirements.txt
```

### 4. Executar o Notebook

Abra o notebook `movie_review_classif.ipynb` em seu editor de código preferido (como VS Code ou Jupyter Lab) e execute as células na ordem.

    Nota: Na primeira execução, a biblioteca kagglehub fará o download do dataset (aprox. 64MB) e os modelos serão baixados (BART-Large tem ~1.6GB e DistilBERT ~268MB). Os checkpoints do modelo treinado serão salvos na pasta ./results.

### 📂 Estrutura do Projeto

<pre>
.
├── .venv/                      # Ambiente virtual (ignorado)
├── results/                    # Checkpoints do modelo Fine-Tuned (gerado)
├── logs/                       # Logs de treino (gerado)
├── movie_review_classif.ipynb  # O notebook principal do projeto
├── requirements.txt            # Dependências de Python
├── other-requirements.txt      # Dependências do PyTorch (GPU)
└── README.md                   # Este arquivo
</pre>