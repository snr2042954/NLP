# Multilingual Sentiment Analysis

Modular framework for comparing multilingual sentiment analysis approaches.

## Runtime

- Python 3.14
- Dependencies are in `requirements.txt`

## Current model design

The codebase now separates embedding generation from classification:

- `models/embedding/`
  - `tfidf.py` — `TFIDFEmbedder`
  - `bert.py` — `BERTEmbedder`
  - `labse.py` — `LaBSEEmbedder`
  - `fasttext.py` — `FastTextEmbedder` (optional)
- `models/classification/`
  - `logistic_regression.py` — `LogisticRegressionClassifier`

This allows easy mixing of any embedder with any classifier.

## Supported combinations

- TF-IDF embeddings + Logistic Regression
- BERT embeddings + Logistic Regression
- LaBSE embeddings + Logistic Regression
- FastText embeddings + Logistic Regression (Windows compatibility is limited)

## FastText on Windows

FastText is not reliably installable on native Windows in many environments. If you are on Windows:

- use `TFIDFEmbedder`, `BERTEmbedder`, or `LaBSEEmbedder`
- or run the project inside WSL / Linux if you need FastText

## Features

- Train on one language, evaluate across multiple languages
- Shared preprocessing and evaluation pipeline
- Modular embedder + classifier architecture

## Usage

Run the main experiment:

```bash
python main.py
```

To change the experiment, edit `main.py` and update the chosen embedder and classifier instances.

Example:

```python
from models.embedding.tfidf import TFIDFEmbedder
from models.classification.logistic_regression import LogisticRegressionClassifier

embedder = TFIDFEmbedder()
classifier = LogisticRegressionClassifier()
```

Then pass them into `ExperimentRunner`.

## Repository structure

```
NLP/
├── models/
│   ├── classification/
│   │   └── logistic_regression.py
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

## Notes

- Models are configured for CPU execution.
- The first run may download pretrained models for BERT and LaBSE.
- Evaluation now suppresses undefined precision warnings by using `zero_division=0` in metric computation.



