"""
main.py
"""

from utils.load_data import load_data
from utils.preprocessing import apply_preprocessing, decode_labels
from utils.evaluation import print_metrics

from models.baseline_tfidf import BaselineTFIDF
from models.BERT import BERTModel
from models.fasttext_model import FastTextModel
from models.labse import LaBSEModel


class ExperimentRunner:
    def __init__(
        self,
        model_name="bert",
        train_language="english",
        languages=("english", "german", "chinese"),
        train_frac=0.1,
        test_frac=0.1,
    ):
        self.model_name = model_name
        self.train_language = train_language
        self.languages = languages
        self.train_frac = train_frac
        self.test_frac = test_frac

        self.model = self._init_model()

    def _init_model(self):
        if self.model_name == "tfidf":
            return BaselineTFIDF()
        elif self.model_name == "bert":
            return BERTModel(epochs=2, batch_size=8)
        elif self.model_name == "fasttext":
            return FastTextModel()
        elif self.model_name == "labse":
            return LaBSEModel()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

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

        # 🔥 CENTRALIZED PREPROCESSING
        self.train_df = apply_preprocessing(train_df)
        self.test_df = apply_preprocessing(test_df)

        print(f"Train size: {len(self.train_df)}")
        print(f"Test size: {len(self.test_df)}")

    def train(self):
        train_subset = self.train_df[
            self.train_df["language"] == self.train_language
        ]

        print(f"\nTraining on: {self.train_language}")
        self.model.train(train_subset)

    def evaluate(self):
        print("\n=== EVALUATION ===")

        results = {}

        for lang in self.languages:
            test_subset = self.test_df[self.test_df["language"] == lang]

            print(f"\nEvaluating on: {lang}")
            metrics = self.model.evaluate(test_subset)

            print_metrics(metrics)
            results[lang] = metrics

        return results

    def run(self):
        print("\n=== CONFIG ===")
        print(f"Model: {self.model_name}")
        print(f"Train language: {self.train_language}")
        print(f"Languages: {self.languages}")

        self.load_data()
        self.train()
        return self.evaluate()


if __name__ == "__main__":
    runner = ExperimentRunner(
        model_name="bert",   # "tfidf" | "bert" | "fasttext" | "labse"
        train_language="english",
        languages=("english", "german", "chinese"),
        train_frac=0.05,
        test_frac=0.05,
    )

    runner.run()