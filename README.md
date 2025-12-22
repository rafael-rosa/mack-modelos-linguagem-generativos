# Projeto: Zero-Shot vs. Fine-Tuning para classificação de Gênero/Tópico a partir de um texto

Este repositório contém o projeto final da disciplina **"Modelos de Linguagem e Generativos"**, focado na comparação de duas técnicas centrais de PLN: Classificação Zero-Shot e Fine-Tuning.

### Programa de pós-graduação em Computação Aplicada - Mestrado Profissional - Mackenzie
* `Prof. Rogério de Oliveira`
### Autores
* `Gildo Manzi da Silva` - RA: 10329658
* `Rafael da Silva Rosa` - RA: 10746329
* `Rogério Goussain Labat` - RA: 10746326

---

## 🎯 Introdução

O objetivo deste projeto é comparar o desempenho de duas abordagens distintas para uma tarefa de análise de texto (determinar o gênero de um filme a partir de sua sinopse) usando um dataset construído com dados do IMDB e do OMDB.

As duas abordagens comparadas são:

1.  **Classificação Zero-Shot (Baseline):** Utiliza um modelo generalista (`facebook/bart-large-mnli`) que classifica o texto sem nunca ter sido treinado especificamente nesta tarefa.
2.  **Fine-Tuning (Desafiante):** Utiliza um modelo especialista (`distilbert-base-uncased`) que é treinado (ajustado) em milhares de exemplos específicos da tarefa. Importante mencionar que **implementamos uma Custom Loss Function** com pesos inversamente proporcionais à frequência das classes para mitigar o viés do modelo em direção às classes majoritárias.

A hipótese inicial é que o modelo Fine-Tuned, por ser especialista, superaria o modelo Zero-Shot generalista.

---

## 🏆 Resultados Principais

A tarefa de classificação multiclasse expôs as limitações da abordagem Zero-Shot e a força do Fine-Tuning:

| MODELO | ESTRATÉGIA | MODELO BASE | ACURÁCIA | F1-SCORE (WEIGHTED) | TEMPO DE INFERÊNCIA | 
| -------- | -------- | ----------- | -------- | ------------------- | ------------------- |
| Baseline | Zero-Shot | ***facebook/bart-large-mnli*** | **15%** | 0.15 | Alto (lento) |
| Desafiante | Fine-Tuning | ***distilbert-base-uncased*** | **42%** | 0.42 | Baixo (Rápido) |

**Análise da Conclusão:**
1.  **O Desafio da Ambiguidade:** Em um cenário com 27 classes possíveis, a fronteira entre gêneros como "Ação", "Aventura" e "Crime" é tênue. O modelo Zero-Shot, por ser generalista, tende a se confundir com a sobreposição de temas. O modelo Fine-Tuned, por outro lado, aprendeu as nuances específicas de como *este dataset* define cada gênero.

2.  **Eficiência Computacional:** A abordagem Zero-Shot exigiu que o modelo processasse cada sinopse comparando-a com todas as etiquetas candidatas, tornando a inferência significativamente mais lenta. O modelo Fine-Tuned (DistilBERT), além de ser arquiteturalmente mais leve (66M vs 406M parâmetros), realiza a classificação em uma única passagem direta (forward pass), sendo ideal para ambientes de produção.

### Veredito Final
Para tarefas complexas de classificação multiclasse com definições de domínio específicas, o **Fine-Tuning é indispensável**. Embora o Zero-Shot seja uma ferramenta poderosa para prototipagem rápida e situações de "cold start" (sem dados), ele não consegue competir com a precisão e a eficiência de um modelo especialista treinado (mesmo que menor) quando dados rotulados estão disponíveis.

---

## 🧠 Referencial Teórico

### Classificação Zero-Shot
A classificação Zero-Shot é uma técnica onde um modelo pode classificar dados em categorias que não viu durante o treinamento. No contexto de PLN, isso é comumente alcançado "reformulando" a tarefa de classificação como uma tarefa de **Inferência de Linguagem Natural (NLI)**. O modelo avalia a probabilidade de uma "premissa" (a sinopse do filme) implicar logicamente uma "hipótese" (ex: "Este filme é um Drama"). O modelo `bart-large-mnli` é pré-treinado na tarefa Multi-Genre NLI (MNLI), tornando-o ideal para isso.

### Fine-Tuning
Fine-Tuning (Ajuste Fino) é o processo de pegar um modelo de fundação pré-treinado (como o `DistilBERT`, que foi treinado para "entender" a linguagem em geral) e treiná-lo um pouco mais em um conjunto de dados específico e com uma tarefa específica. Uma "cabeça de classificação" é adicionada ao topo do modelo, e seus pesos são ajustados para se especializarem na nova tarefa (neste caso, determinar o gênero do filme a partir do plot).

---

## 🧪 Metodologia

O projeto foi conduzido em três etapas principais:

### 1. Coleta de dados e preparação dos dados
* Download do dataset com ID's dos filmes no site do IMDB: https://datasets.imdbws.com/
* Coleta das sinopses e gêneros dos filmes via API do OMDB: https://www.omdbapi.com/
* Compilação de um dataset único com os dados coletados
* Remoção de nulos e normalização dos dados

### 2. Baseline: Modelo Zero-Shot
* **Modelo:** `facebook/bart-large-mnli` (406M de parâmetros).
* **Processo:** 500 sinopses do dataset foram selecionadas aleatoriamente.
* **Execução:** O modelo `pipeline` foi carregado na GPU (`cuda:0`) e classificou as 500 amostras com 27 rótulos possíveis `['DOCUMENTARY', 'COMEDY', 'DRAMA', 'SHORT', 'WESTERN', 'THRILLER', 'ANIMATION', 'MUSIC', 'CRIME', 'SCI-FI', 'HORROR', 'TALK-SHOW', 'FAMILY', 'ACTION', 'MYSTERY', 'BIOGRAPHY', 'REALITY-TV', 'NEWS', 'FANTASY', 'ROMANCE', 'MUSICAL', 'SPORT', 'HISTORY', 'GAME-SHOW', 'ADVENTURE', 'WAR', 'ADULT']`.
* **Métricas:** Um `classification_report` (Acurácia, Precisão, Recall, F1-Score) foi gerado comparando as previsões com os rótulos reais.

### 3. Desafiante: Modelo Fine-Tuned
* **Modelo:** `distilbert-base-uncased` (66M de parâmetros).
* **Processo:**
    1.  Uma amostra de 1500 plots foi selecionada do dataset.
    2.  Os dados foram divididos em 60% para treino e 40% para teste.
    3.  O `AutoTokenizer` foi usado para preparar os dados.
    4.  O modelo foi treinado por 10 épocas usando um `Trainer` customizado com suporte a função de perdas.
    5.  Foi configurado para salvar apenas o melhor modelo (`load_best_model_at_end=True`) evitando overfitting.
* **Métricas:** As mesmas métricas foram calculadas no conjunto de teste.

---

## 🚀 Instruções de Uso

### Pré-requisitos
* Python 3.13+
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

Abra o notebook `movie_plot_classifier.ipynb` em seu editor de código preferido (como VS Code ou Jupyter Lab) e execute as células na ordem.

    Nota: Na primeira execução, os modelos serão baixados (BART-Large tem ~1.6GB e DistilBERT ~268MB) e os checkpoints do modelo treinado serão salvos na pasta ./results.

### ⚠️ IMPORTANTE: Execução via Google Colab

Executar o notebook via Google Colab poderá exigir o fornecimento de uma API Key do `wandb.ai` no passo **3.2 (Treinamento)**. Obtenha uma API Key criando uma conta em https://wandb.ai/authorize?ref=models e forneça a chave diretamente no Colab no momento em que for solicitada. **O não fornecimento da API Key implicará em erro de execução.**



### 📂 Estrutura do Projeto

<pre>
.
├── .venv/                          # Ambiente virtual (ignorado)
├── coleta_dados/                   # Processos de coleta de dados
    ├── movie_plot_gathering.ipynb  # Notebok para ler IDs dos filmes e coletar dados via API do OMDB
    ├── /imdb_dataset/  
        └── title.ratings.tsv       # Dataset com IDs de filmes do IMDB
    └── /out/                       # Dados dos filmes coletados via API do OMDB (porções de mil)
├── data_prep/
    └── create_movies_dataset.ipynb # Compila os dados coletados e um único dataset e faz a preparação dos dados
├── movies_dataset/
    └── movie_plots_dataset.csv     # Dataset final para uso nos modelos
├── results/                        # Checkpoints do modelo Fine-Tuned (gerado)
├── logs/                           # Logs de treino (gerado)
├── movie_plot_classifier.ipynb     # O notebook principal do projeto
├── requirements.txt                # Dependências de Python
├── other-requirements.txt          # Dependências do PyTorch (GPU)
└── README.md                       # Este arquivo
</pre>


---

# 💲Aplicações práticas reais - Plano de negócios

Duas sugestões de aplicações possíveis para o projeto, conectando diretamente as tecnologias objetos do teste (***Zero-Shot e Fine-Tuning***) com cenários de negócio tangíveis:

## 1️⃣. Cenário "Streaming & Mídia": ***Catalogação Automática de Conteúdo (Metadata Tagging)***

Este é o uso mais direto do seu dataset de filmes, mas aplicado a plataformas como Globoplay, Spotify ou Marketplaces de E-books (Kindle).

**O Problema:** Uma plataforma recebe milhares de novos conteúdos (vídeos de parceiros, podcasts, livros indie) por dia. Classificar manualmente se um vídeo é "Esportes", "E-Sports" ou "Lazer" é lento e caro. Além disso, surgem novos gêneros o tempo todo (ex: "True Crime" não era uma categoria forte há 10 anos).

**A Aplicação Híbrida:**

+ **Fine-Tuning**: O modelo treinado varre todo o catálogo existente e novos uploads diários, classificando-os rapidamente nas categorias "pai" (Ação, Drama, Comédia). Isso garante velocidade e baixo custo de nuvem.
+ **Zero-Shot**: A equipe de marketing quer criar uma coleção temporária para o Halloween ou para uma tendência do TikTok (ex: "Dark Academia"). Eles não têm dados para treinar um modelo. Eles usam o Zero-Shot para re-classificar o conteúdo apenas buscando essa tag específica.

#### 📈 **Valor de Negócio:** 

+ `Redução` de custo operacional (menos humanos tagueando)
+ `+ Agilidade` de Marketing (criar vitrines temáticas em minutos, não semanas).

## 2️⃣. Cenário "Atendimento ao Cliente": ***Roteamento Inteligente de Tickets (Smart Triage)***

Trocamos "Sinopse do Filme" por "Descrição do Problema do Cliente" e "Gênero" por "Departamento Responsável.

**O Problema:** Uma empresa de Telecom ou um Banco recebe milhares de e-mails/chamados por dia. Atualmente, um humano lê e decide: "Isso vai para o Financeiro", "Isso é Suporte Técnico", "Isso é Vendas". Esse humano é um gargalo..

**A Aplicação:**

+ `Fine-Tuning (Alta Eficiência)`: O modelo é treinado com o histórico de chamados dos últimos 2 anos. Exemplo: `"Minha fatura veio cobrando o valor errado." → Modelo prevê: Financeiro (99% confiança).`

#### 📈 **Valor de Negócio:** 

+ `Fine-Tuning:` Automatiza 80-90% da triagem (reduzindo tempo de resposta de horas para segundos).
