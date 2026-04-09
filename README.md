# Multilingual Sentiment Analysis

Simple framework to compare different models for multilingual sentiment classification.

## Runtime

python3.14
dependencies in requirements.txt

## Models
- TF-IDF + Logistic Regression
- BERT (bert-base-multilingual-cased)
- LaBSE + Logistic Regression
- fasttext (not working yet)

## Features
- Train on one language, evaluate on multiple
- Shared preprocessing and evaluation
- Easy model switching via config

## Usage

Run the main pipeline:

```bash
python main.py
```

Edit main.py to change:

* model (tfidf, bert, labse)
* training language
* dataset fraction

## Structure
```
project/
├── models/
│   ├── data/                  # (ignored) downloaded model files from fasttext
│   ├── baseline_tfidf.py
│   ├── BERT.py
│   ├── fasttext_model.py
│   └── labse.py
│
├── utils/
│   ├── evaluation.py
│   ├── load_data.py
│   └── preprocessing.py
│
├── main.py
├── requirements.txt
├── README.md
```

## Notes

Models run on CPU (GPU not supported in this setup)
First run downloads pretrained models (BERT / LaBSE)





## Models

### Embedding

* BERT
* Fasttext
* Tfidf
* LaBSE

### Classifier

* Bidirectional LSTM
* SVM
* logistic regression

