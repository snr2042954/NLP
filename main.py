"""
main.py
"""

from utils.load_data import load_data
from utils.preprocessing import apply_preprocessing, decode_labels
from utils.evaluation import print_metrics, compute_metrics

from models.embedding.tfidf import TFIDFEmbedder
from models.embedding.bert import BERTEmbedder
from models.embedding.labse import LaBSEEmbedder
from models.classification.logistic_regression import LogisticRegressionClassifier

# FastText is not compatible with Windows
try:
    from models.embedding.fasttext import FastTextEmbedder
    FASTTEXT_AVAILABLE = True
except ImportError:
    FastTextEmbedder = None
    FASTTEXT_AVAILABLE = False


class ExperimentRunner:
    def __init__(
        self,
        embedder,
        classifier,
        train_language="english",
        languages=("english", "german", "chinese"),
        train_frac=0.1,
        test_frac=0.1,
    ):
        self.embedder = embedder
        self.classifier = classifier
        self.train_language = train_language
        self.languages = languages
        self.train_frac = train_frac
        self.test_frac = test_frac

    def load_data(self):
        print("Loading data...")

        train_df = load_data(
            languages=self.languages,
            split="train",
            frac=self.train_frac,
        )

        test_df = load_data(
            languages=self.languages,
            split="test",
            frac=self.test_frac,
        )

        # CENTRALIZED PREPROCESSING
        self.train_df = apply_preprocessing(train_df)
        self.test_df = apply_preprocessing(test_df)

        print(f"Train size: {len(self.train_df)}")
        print(f"Test size: {len(self.test_df)}")

    def train(self):

        # Subset the training data
        train_subset = self.train_df[
            self.train_df["language"] == self.train_language
        ]
        train_texts = train_subset["text"].astype(str).tolist()
        train_labels = train_subset["label"].values
        print(f"\nTraining on: {self.train_language}")

        # Fit embedder if needed
        if hasattr(self.embedder, 'fit'):
            self.embedder.fit(train_texts)

        # Get embeddings (handle FastText separately since it needs language info)
        if FASTTEXT_AVAILABLE and isinstance(self.embedder, FastTextEmbedder):
            X_train = self.embedder.transform(train_texts, self.train_language)
        else:
            X_train = self.embedder.transform(train_texts)

        # Fit classifier
        self.classifier.fit(X_train, train_labels)

    def evaluate(self):
        print("\n=== EVALUATION ===")

        results = {}

        for lang in self.languages:
            test_subset = self.test_df[self.test_df["language"] == lang]

            print(f"\nEvaluating on: {lang}")

            test_texts = test_subset["text"].astype(str).tolist()
            y_true = test_subset["label"].values

            # Get embeddings (handle FastText separately since it needs language info)
            if FASTTEXT_AVAILABLE and isinstance(self.embedder, FastTextEmbedder):
                X_test = self.embedder.transform(test_texts, lang)
            else:
                X_test = self.embedder.transform(test_texts)

            # Predict
            y_pred = self.classifier.predict(X_test)

            # Compute metrics
            metrics = compute_metrics(y_true, y_pred)

            print_metrics(metrics)
            results[lang] = metrics

        return results

    def run(self):
        print("\n=== CONFIG ===")
        print(f"Embedder: {type(self.embedder).__name__}")
        print(f"Classifier: {type(self.classifier).__name__}")
        print(f"Train language: {self.train_language}")
        print(f"Languages: {self.languages}")

        self.load_data()
        self.train()
        return self.evaluate()


if __name__ == "__main__":

    embedder = BERTEmbedder() # TFIDFEmbedder() | BERTEmbedder() | LaBSEEmbedder() | FastTextEmbedder()
    classifier = LogisticRegressionClassifier()

    runner = ExperimentRunner(
        embedder=embedder,
        classifier=classifier,
        train_language="english",
        languages=("english", "german", "arabic", "portuguese"),
        train_frac=0.05,
        test_frac=0.05,
    )

    runner.run()