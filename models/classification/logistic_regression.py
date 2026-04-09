"""
models/classification/logistic_regression.py

Logistic Regression classifier for sentiment analysis.
"""

from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier:
    def __init__(
        self,
        C=1.0,
        random_state=42,
    ):
        """
        Parameters
        ----------
        C : float
            Inverse regularization strength
        random_state : int
            Seed for reproducibility
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the Logistic Regression model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (embeddings)
        y : np.ndarray
            Labels
        """
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Predict labels.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (embeddings)

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (embeddings)

        Returns
        -------
        np.ndarray
            Predicted probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict_proba(X)