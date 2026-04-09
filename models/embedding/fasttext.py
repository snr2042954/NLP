"""
models/embedding/fasttext.py

FastText embedder for multilingual sentiment analysis.
"""

import os
import gzip
import shutil
import urllib.request
import numpy as np
import fasttext

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


class FastTextEmbedder:
    def __init__(self):
        self.models = {}

    def _get_ft_model(self, lang):
        if lang not in self.models:
            if lang not in _FT_MODEL_CACHE:
                code = LANG_CODES[lang]
                path = download_model(code)
                _FT_MODEL_CACHE[lang] = fasttext.load_model(path)
            self.models[lang] = _FT_MODEL_CACHE[lang]
        return self.models[lang]

    def transform(self, texts, lang):
        """
        Transform texts to FastText embeddings.

        Parameters
        ----------
        texts : list of str
            Texts to embed (should be preprocessed)
        lang : str
            Language of the texts

        Returns
        -------
        np.ndarray
            FastText embeddings
        """
        ft = self._get_ft_model(lang)
        return np.array([ft.get_sentence_vector(t) for t in texts])