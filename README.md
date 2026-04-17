# Multilingual Sentiment Analysis

Modular framework for comparing multilingual sentiment analysis approaches.

## Overview

This repository separates embedding generation from classification so you can compare combinations of multilingual embedders and classifiers.

## Runtime

- Python 3.14
- Dependencies: `requirements.txt`

## Components

- `main.py` — runs experiments and saves `results.csv`
- `models/embedding/` — embedder implementations
  - `tfidf.py` — TF-IDF
  - `bert.py` — BERT
  - `labse.py` — LaBSE
  - `fasttext.py` — FastText (optional)
- `models/classification/` — classifier implementations
  - `logistic_regression.py`
  - `mlp.py`
  - `xlmr_head.py`
- `utils/` — shared utilities
  - `load_data.py`
  - `preprocessing.py`
  - `evaluation.py`

## Behavior

`main.py` trains on one language and evaluates across multiple languages. It prints metrics and writes a summary CSV to `results.csv`, which can be opened in Excel.

## Usage

Run the main experiment:

```bash
python main.py
```

To customize experiments, edit `main.py` and swap embedders/classifiers.

## Repository structure

```
NLP/
├── _multilingual-sentiments/
├── models/
│   ├── classification/
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
├── requirements.txt
└── README.md
```



