"""
models/fasttext_model.py

FastText + Logistic Regression baseline.

Pipeline:
- FastText sentence embeddings
- Logistic Regression classifier
"""

import os
import gzip
import shutil
import urllib.request
import numpy as np
import fasttext

from sklearn.linear_model import LogisticRegression

from utils.evaluation import compute_metrics
from utils.preprocessing import apply_preprocessing


LANG_CODES = {
    "english": "en",
    "german": "de",
    "arabic": "ar",
    "portuguese": "pt",
}

FASTTEXT_DIR = "data/fasttext_models"
_FT_MODEL_CACHE = {}


def download_model(lang_code):
    os.makedirs(FASTTEXT_DIR, exist_ok=True)

    bin_path = os.path.join(FASTTEXT_DIR, f"cc.{lang_code}.300.bin")
    gz_path = bin_path + ".gz"

    if os.path.exists(bin_path):
        return bin_path

    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang_code}.300.bin.gz"
    print(f"Downloading FastText model for {lang_code}...")

    urllib.request.urlretrieve(url, gz_path)

    with gzip.open(gz_path, "rb") as f_in, open(bin_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    return bin_path


class FastTextModel:
    def __init__(self):
        self.model = None
        self.train_language = None

    def _get_ft_model(self, lang):
        if lang not in _FT_MODEL_CACHE:
            code = LANG_CODES[lang]
            path = download_model(code)
            _FT_MODEL_CACHE[lang] = fasttext.load_model(path)
        return _FT_MODEL_CACHE[lang]

    def _vectorize(self, df, lang):
        ft = self._get_ft_model(lang)
        return np.array([ft.get_sentence_vector(t) for t in df["text_clean"]])

    def train(self, train_df):
        train_df = apply_preprocessing(train_df)

        self.train_language = train_df["language"].iloc[0]

        X = self._vectorize(train_df, self.train_language)
        y = train_df["label"].values

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

    def predict(self, test_df):
        test_df = apply_preprocessing(test_df)

        lang = test_df["language"].iloc[0]
        X = self._vectorize(test_df, lang)

        return self.model.predict(X)

    def evaluate(self, test_df):
        y_true = test_df["label"].values
        y_pred = self.predict(test_df)

        return compute_metrics(y_true, y_pred)

if __name__ == "__main__":

    from utils.load_data import load_data
    from utils.preprocessing import apply_preprocessing
    from utils.evaluation import print_metrics

    print("Loading data...")
    languages = ("english", "german", "arabic", "portuguese")

    train_df = load_data(languages=languages, split="train", frac=0.02)
    test_df = load_data(languages=languages, split="test", frac=0.02)

    # Apply preprocessing (IMPORTANT — now centralized)
    train_df = apply_preprocessing(train_df)
    test_df = apply_preprocessing(test_df)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    train_lang = "english"
    train_subset = train_df[train_df["language"] == train_lang]

    print(f"\nTraining FastText on: {train_lang}")
    model = FastTextModel()
    model.train(train_subset)

    for lang in languages:
        print(f"\nEvaluating on: {lang}")
        test_subset = test_df[test_df["language"] == lang]

        metrics = model.evaluate(test_subset)
        print_metrics(metrics)

    print("\nDone.")