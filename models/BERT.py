"""
models/BERT.py

Multilingual BERT model for sentiment analysis.

Pipeline:
- Tokenization (bert-base-multilingual-cased)
- Transformer fine-tuning
"""

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

from utils.evaluation import compute_metrics

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class BERTModel:
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        num_labels=3,
        lr=2e-5,
        batch_size=16,
        epochs=2,
    ):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        ).to(self.device)

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

        self.is_fitted = False

    def train(self, train_df):
        """
        Train BERT model.
        """

        texts = train_df["text"].astype(str).tolist()
        labels = train_df["label"].tolist()

        dataset = TextDataset(texts, labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0

            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        self.is_fitted = True

    def predict(self, test_df):
        """
        Predict labels.
        """

        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        texts = test_df["text"].astype(str).tolist()

        dataset = TextDataset(texts, [0] * len(texts), self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}

                outputs = self.model(**batch)
                logits = outputs.logits

                batch_preds = torch.argmax(logits, dim=1)
                preds.extend(batch_preds.cpu().numpy())

        return preds

    def evaluate(self, test_df):
        """
        Evaluate model.
        """

        y_true = test_df["label"].values
        y_pred = self.predict(test_df)

        return compute_metrics(y_true, y_pred)


if __name__ == "__main__":

    from utils.load_data import load_data
    from utils.evaluation import print_metrics

    print("Loading data...")
    train_df = load_data(split="train", frac=0.05)
    test_df = load_data(split="test", frac=0.05)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    train_lang = "english"
    train_subset = train_df[train_df["language"] == train_lang]

    print(f"\nTraining BERT on: {train_lang}")
    model = BERTModel(epochs=2)  # few epochs as requested
    model.train(train_subset)

    for lang in test_df["language"].unique():
        print(f"\nEvaluating on: {lang}")
        test_subset = test_df[test_df["language"] == lang]

        metrics = model.evaluate(test_subset)
        print_metrics(metrics)

    print("\nDone.")