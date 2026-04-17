"""
main.py
"""

from utils.load_data import load_data
from utils.preprocessing import apply_preprocessing, decode_labels
from utils.evaluation import print_metrics, compute_metrics

import csv
import os

from models.embedding.tfidf import TFIDFEmbedder
from models.embedding.bert import BERTEmbedder
from models.embedding.labse import LaBSEEmbedder
from models.classification.logistic_regression import LogisticRegressionClassifier
from models.classification.mlp import MLPClassifier
from models.classification.xlmr_head import XLMRClassifierHead

# FastText is not compatible with Windows
try:
    from models.embedding.fasttext import FastTextEmbedder
    FASTTEXT_AVAILABLE = True
except ImportError:
    FastTextEmbedder = None
    FASTTEXT_AVAILABLE = False


class SingleExperimentRunner:
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


def save_results_to_csv(all_results, filepath="results.csv"):
    fieldnames = [
        "embedder",
        "classifier",
        "train_language",
        "test_language",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (embedder_name, classifier_name, train_language), results in all_results.items():
            for test_language, metrics in results.items():
                row = {
                    "embedder": embedder_name,
                    "classifier": classifier_name,
                    "train_language": train_language,
                    "test_language": test_language,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                }
                writer.writerow(row)


def run_all(test_frac,train_frac):
    """This function calls the main experiment runners:
    it checks all combinations of embedders and classifiers and saves the results into a dictionary.
    for every embedder-classifier combination it takes as the train language each of the languages in the dataset and evaluates on all languages.
    """
    embedders = [
        TFIDFEmbedder(),
        BERTEmbedder(),
        LaBSEEmbedder(),
    ]

    if FASTTEXT_AVAILABLE:
        embedders.append(FastTextEmbedder())

    classifiers = [
        LogisticRegressionClassifier(),
        MLPClassifier(),
        XLMRClassifierHead(),
    ]

    all_results = {}

    for embedder in embedders:
        for classifier in classifiers:
            for train_lang in ["english", "german", "arabic", "portuguese"]:
                print(f"\nRunning experiment with {type(embedder).__name__} + {type(classifier).__name__} (train language: {train_lang})")
                runner = SingleExperimentRunner(
                    embedder=embedder,
                    classifier=classifier,
                    train_language=train_lang,
                    languages=("english", "german", "arabic", "portuguese"),
                    train_frac=train_frac,
                    test_frac=test_frac,
                )
                results = runner.run()
                all_results[(type(embedder).__name__, type(classifier).__name__, train_lang)] = results

    return all_results


if __name__ == "__main__":

    # CONGIGURABLES
    TEST_FRACTION = 0.2
    TRAIN_FRACTION = 0.2
    OUTPUT_PATH = "results.csv"

    # Run all experiments and save results to CSV
    results = run_all(test_frac=TEST_FRACTION, train_frac=TRAIN_FRACTION)
    save_results_to_csv(results, OUTPUT_PATH)
    print(f"\nSaved results to {os.path.abspath(OUTPUT_PATH)}")