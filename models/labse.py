"""
models/labse.py

LaBSE embeddings + Logistic Regression classifier.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from utils.evaluation import compute_metrics


class LaBSEModel:
    def __init__(
        self,
        model_name="sentence-transformers/LaBSE",
        C=1.0,
    ):
        print("Loading LaBSE model...")
        self.encoder = SentenceTransformer(model_name)

        self.clf = LogisticRegression(max_iter=1000, C=C)
        self.is_fitted = False

    def _encode(self, texts):
        return self.encoder.encode(texts, show_progress_bar=True)

    def train(self, train_df):
        texts = train_df["text"].astype(str).tolist()
        labels = train_df["label"].values

        X = self._encode(texts)
        self.clf.fit(X, labels)

        self.is_fitted = True

    def predict(self, test_df):
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        texts = test_df["text"].astype(str).tolist()
        X = self._encode(texts)

        return self.clf.predict(X)

    def evaluate(self, test_df):
        y_true = test_df["label"].values
        y_pred = self.predict(test_df)

        return compute_metrics(y_true, y_pred)


# -------------------------
# QUICK TEST
# -------------------------

if __name__ == "__main__":
    from utils.load_data import load_data
    from utils.preprocessing import apply_preprocessing
    from utils.evaluation import print_metrics

    languages = ("english", "german", "arabic", "portuguese")

    print("Loading data...")
    train_df = load_data(languages=languages, split="train", frac=0.05)
    test_df = load_data(languages=languages, split="test", frac=0.05)

    train_df = apply_preprocessing(train_df)
    test_df = apply_preprocessing(test_df)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    train_lang = "english"
    train_subset = train_df[train_df["language"] == train_lang]

    print(f"\nTraining LaBSE on: {train_lang}")
    model = LaBSEModel()
    model.train(train_subset)

    for lang in languages:
        print(f"\nEvaluating on: {lang}")
        test_subset = test_df[test_df["language"] == lang]

        metrics = model.evaluate(test_subset)
        print_metrics(metrics)

    print("\nDone.")