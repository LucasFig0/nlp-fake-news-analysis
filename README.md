# Discourse Analysis and Misinformation Detection: Classical NLP vs. Zero-Shot LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucasFig0/nlp-fake-news-analysis/blob/main/nlp_fake_news_analysis.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end computational social science and natural language processing (NLP) project that investigates textual patterns in digital media, benchmarking supervised machine learning pipelines against pre-trained Large Language Model (LLM) zero-shot inference.

---

## Motivation & The Critical Role of Misinformation Detection

In modern digital ecosystems, the velocity and volume of online information flows have amplified the spread of coordinated disinformation campaigns, algorithmic polarization, and misleading narratives. The erosion of public trust in factual reporting poses severe challenges to democratic institutions, public health communication, and social cohesion.

Automated misinformation detection is no longer solely a classification exercise; it is an essential instrument for computational social scientists and engineers striving to:
* **Quantify Rhetorical Framing:** Identify stylistic markers, sensationalism, and discourse shifts at scale.
* **Support Real-Time Moderation:** Process streaming data feeds with low latency and high statistical precision.
* **Evaluate AI Trade-offs:** Balance high-capacity foundation models with lightweight, resource-efficient statistical learners.

---

## Project Objectives

* **Discourse & Lexical Analysis:** Extract n-grams, vocabulary distributions, and lexical patterns distinguishing verifiable journalism from unverified claims.
* **Supervised Pipeline Benchmarking:** Build and optimize end-to-end Scikit-Learn pipelines pairing TF-IDF feature extraction with Logistic Regression and Multinomial Naive Bayes.
* **Zero-Shot LLM Evaluation:** Assess the out-of-the-box classification capability of a pre-trained Transformer (`facebook/bart-large-mnli`) without domain-specific fine-tuning.
* **Engineering Trade-off Assessment:** Measure inference latency, computational cost, and macro F1-scores to guide production and research deployments.

---

## Core Tech Stack & Tooling

* **Data Engineering & Manipulation:** `Pandas`, `NumPy`
* **Text Preprocessing & NLP:** `NLTK` (stopwords, tokenization), `Regex`
* **Exploratory Data Analysis & Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning Pipelines:** `Scikit-Learn` (`TfidfVectorizer`, `Pipeline`, `LogisticRegression`, `MultinomialNB`, `metrics`)
* **Deep Learning & Transformers:** `Hugging Face Transformers` (`pipeline("zero-shot-classification")`), `PyTorch`, `Datasets`

---

## Methodology & Pipeline Architecture

```text
Raw Corpus (ISOT/WELFake) 
   │
   ├──► Text Preprocessing (Regex noise removal, lowercasing, stopword stripping)
   │
   ├──► Exploratory Data Analysis (Class balance & top n-grams extraction)
   │
   ├──► Stratified Split (80% Train / 20% Test)
   │       │
   │       ├──► Supervised Pipeline 1: TF-IDF (1-2 N-grams) + Logistic Regression
   │       └──► Supervised Pipeline 2: TF-IDF (1-2 N-grams) + Multinomial Naive Bayes
   │
   └──► Zero-Shot Pipeline: facebook/bart-large-mnli (Inference on test holdout)

## Empirical Benchmarks

All models were evaluated on a held-out test split under identical conditions:

| Model Architecture | Approach | Accuracy | Macro F1-Score | Inference Latency (50 texts) | Hardware Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression + TF-IDF** | Supervised Baseline | **96.0%** | **0.96** | **130.36 ms** | CPU |
| **Multinomial Naive Bayes + TF-IDF** | Supervised Baseline | 92.7% | 0.93 | 110.15 ms | CPU |
| **Zero-Shot LLM (`BART-Large-MNLI`)** | Foundation Model | 62.0% | 0.61 | 199.04 s | GPU Recommended |

---

## Key Takeaways & Technical Insights

* **Linguistic Signatures:** Factual reporting exhibits consistent institutional sourcing patterns (`said`, `reuters`, `state`, `government`), whereas disinformation leans on informal syntax, emotional appeals, and targeted political entities (`trump`, `people`, `like`, `obama`).
* **Supervised vs. Zero-Shot Performance:** Supervised linear models heavily outperformed off-the-shelf zero-shot inference on this domain. Without few-shot conditioning or fine-tuning, generalized NLI models struggle with domain-specific journalistic phrasing.
* **Latency & Production Trade-offs:** The classical TF-IDF + Logistic Regression pipeline achieved a **~1,500x speedup** over the Transformer model, proving significantly more viable for high-throughput digital media streams.
* **Model Interpretability:** Extracting learned TF-IDF coefficients provided clear visibility into decision boundaries, offering qualitative auditing capabilities essential for computational social science research.

---

## Getting Started

### Local Setup

```bash
# 1. Clone the repository
git clone [https://github.com/LucasFig0/nlp-fake-news-analysis.git](https://github.com/LucasFig0/nlp-fake-news-analysis.git)
cd nlp-fake-news-analysis

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
