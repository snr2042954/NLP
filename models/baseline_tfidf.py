"""
models/baseline_tfidf.py

TF-IDF baseline model for multilingual sentiment analysis.

Pipeline:
- TF-IDF vectorization
- Logistic Regression classifier
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.evaluation import compute_metrics


class BaselineTFIDF:
    def __init__(
        self,
        max_features=5000,
        ngram_range=(1, 2),
        C=1.0,
        random_state=42,
    ):
        """
        Parameters
        ----------
        max_features : int
            Maximum number of TF-IDF features
        ngram_range : tuple
            N-gram range for vectorizer
        C : float
            Inverse regularization strength for LogisticRegression
        random_state : int
            Seed for reproducibility
        """

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

        self.model = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=random_state,
        )

        self.is_fitted = False

    def train(self, train_df):
        """
        Train model on training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain columns: text, label
        """

        X = train_df["text"].astype(str).values
        y = train_df["label"].values

        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

        self.is_fitted = True

    def predict(self, test_df):
        """
        Predict labels for test data.

        Parameters
        ----------
        test_df : pd.DataFrame

        Returns
        -------
        list
            Predicted labels
        """

        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        X = test_df["text"].astype(str).values
        X_vec = self.vectorizer.transform(X)

        return self.model.predict(X_vec)

    def evaluate(self, test_df):
        """
        Evaluate model using shared evaluation utilities.

        Parameters
        ----------
        test_df : pd.DataFrame

        Returns
        -------
        dict
            Metrics dictionary
        """

        y_true = test_df["label"].values
        y_pred = self.predict(test_df)

        return compute_metrics(y_true, y_pred)


if __name__ == "__main__":

    from utils.load_data import load_data
    from utils.evaluation import print_metrics

    print("Loading data...")
    train_df = load_data(split="train", frac=0.1)  # small subset for quick test
    test_df = load_data(split="test", frac=0.1)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    # Train model on English only (example baseline setup)
    train_lang = "english"
    train_subset = train_df[train_df["language"] == train_lang]

    print(f"\nTraining on language: {train_lang}")
    model = BaselineTFIDF()
    model.train(train_subset)

    # Evaluate on all languages
    for lang in test_df["language"].unique():
        print(f"\nEvaluating on: {lang}")
        test_subset = test_df[test_df["language"] == lang]

        metrics = model.evaluate(test_subset)
        print_metrics(metrics)

    print("\nDone.")