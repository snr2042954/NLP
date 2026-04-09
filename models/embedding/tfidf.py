"""
models/embedding/tfidf.py

TF-IDF embedder for multilingual sentiment analysis.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFEmbedder:
    def __init__(
        self,
        max_features=5000,
        ngram_range=(1, 2),
    ):
        """
        Parameters
        ----------
        max_features : int
            Maximum number of TF-IDF features
        ngram_range : tuple
            N-gram range for vectorizer
        """

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

        self.is_fitted = False

    def fit(self, texts):
        """
        Fit the TF-IDF vectorizer on the training texts.

        Parameters
        ----------
        texts : list of str
            Training texts
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def transform(self, texts):
        """
        Transform texts to TF-IDF embeddings.

        Parameters
        ----------
        texts : list of str
            Texts to embed

        Returns
        -------
        np.ndarray
            TF-IDF embeddings
        """
        if not self.is_fitted:
            raise RuntimeError("Embedder must be fitted before transforming.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """
        Fit and transform in one step.

        Parameters
        ----------
        texts : list of str
            Texts to embed

        Returns
        -------
        np.ndarray
            TF-IDF embeddings
        """
        self.fit(texts)
        return self.transform(texts)