"""
models/embedding/labse.py

LaBSE embedder for multilingual sentiment analysis.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class LaBSEEmbedder:
    def __init__(
        self,
        model_name="sentence-transformers/LaBSE",
    ):
        """
        Parameters
        ----------
        model_name : str
            Pre-trained LaBSE model name
        """
        print("Loading LaBSE model...")
        self.encoder = SentenceTransformer(model_name, device="cpu")

    def transform(self, texts):
        """
        Transform texts to LaBSE embeddings.

        Parameters
        ----------
        texts : list of str
            Texts to embed

        Returns
        -------
        np.ndarray
            LaBSE embeddings
        """
        return self.encoder.encode(texts, show_progress_bar=True)