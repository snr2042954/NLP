# Multilingual Sentiment Analysis

## Purpose

This project implements a modular framework for multilingual sentiment analysis, enabling the comparison of various embedding techniques and classification models across multiple languages. The framework separates embedding generation from classification, allowing users to experiment with different combinations of multilingual embedders and classifiers. It trains on one language and evaluates performance across multiple languages, providing insights into cross-lingual transfer capabilities.

The project uses the multilingual-sentiments dataset, which contains sentiment-labeled text in multiple languages including English, German, and Chinese.

## Features

- Modular design: Easily swap embedders and classifiers
- Multilingual evaluation: Train on one language, test on multiple
- Comprehensive metrics: Accuracy, precision, recall, F1-score
- Visualization support: Generate plots and figures from results
- Pre-generated results: Includes `results.csv` and `figures/` for quick reference

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd NLP
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Runtime Requirements

- Python 3.14
- Dependencies listed in `requirements.txt`

## Usage

### Running the Main Experiment

To run the experiments and generate `results.csv`:

```bash
python main.py
```

This will:
- Load the multilingual-sentiments dataset
- Apply preprocessing
- Generate embeddings using various embedders (TF-IDF, BERT, LaBSE, FastText if available)
- Train classifiers (Logistic Regression, MLP, XLM-R Head)
- Evaluate on multiple languages
- Save results to `results.csv`

### Generating Visualizations

After running the main experiment (or if using pre-generated results), generate visualizations:

```bash
python visualizations.py
```

This creates plots and figures in the `figures/` directory.

**Note:** The repository already includes `results.csv` and pre-generated images in `figures/` for reference. Running `visualizations.py` is optional and primarily for reproduction or customization.

### Customizing Experiments

To modify experiments:
- Edit `main.py` to change languages, fractions, or model combinations
- Swap embedders/classifiers by importing different classes

## Components

- `main.py` — Main script to run experiments and save `results.csv`
- `visualizations.py` — Script to generate plots from results
- `models/embedding/` — Embedder implementations
  - `tfidf.py` — TF-IDF vectorization
  - `bert.py` — BERT embeddings
  - `labse.py` — LaBSE (Language-agnostic BERT Sentence Embedding)
  - `fasttext.py` — FastText embeddings (optional, may not be compatible on all systems)
- `models/classification/` — Classifier implementations
  - `logistic_regression.py` — Logistic Regression
  - `mlp.py` — Multi-Layer Perceptron
  - `xlmr_head.py` — XLM-R classification head
- `utils/` — Shared utilities
  - `load_data.py` — Data loading functions
  - `preprocessing.py` — Text preprocessing
  - `evaluation.py` — Metrics computation
- `_multilingual-sentiments/` — Dataset directory
- `figures/` — Generated visualization images
- `results.csv` — Experiment results summary

## Repository Structure

```
NLP/
├── _multilingual-sentiments/
│   ├── dataset_infos.json
│   ├── multilingual-sentiments.py
│   └── README_dataset.md
├── figures/
├── models/
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── mlp.py
│   │   └── xlmr_head.py
│   └── embedding/
│       ├── bert.py
│       ├── fasttext.py
│       ├── labse.py
│       └── tfidf.py
├── utils/
│   ├── evaluation.py
│   ├── load_data.py
│   └── preprocessing.py
├── main.py
├── visualizations.py
├── requirements.txt
├── results.csv
└── README.md
```

## Results

The `results.csv` file contains evaluation metrics for each embedder-classifier combination across languages. It can be opened in Excel or analyzed programmatically.

Pre-generated figures in `figures/` provide visual comparisons of model performance.



