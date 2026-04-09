"""
models/embedding/bert.py

BERT embedder for multilingual sentiment analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModel


class BERTEmbedder:
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        max_length=128,
        device="cpu",
    ):
        """
        Parameters
        ----------
        model_name : str
            Pre-trained BERT model name
        max_length : int
            Maximum sequence length
        device : str
            Device to run the model on
        """
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

        self.model.eval()  # Set to eval mode for inference

    def _encode_texts(self, texts):
        """
        Tokenize texts.

        Parameters
        ----------
        texts : list of str
            Texts to tokenize

        Returns
        -------
        dict
            Tokenized inputs
        """
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def transform(self, texts):
        """
        Transform texts to BERT embeddings.

        Parameters
        ----------
        texts : list of str
            Texts to embed

        Returns
        -------
        np.ndarray
            BERT embeddings (pooled output)
        """
        inputs = self._encode_texts(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooled output (CLS token) as embedding
            embeddings = outputs.pooler_output.cpu().numpy()

        return embeddings