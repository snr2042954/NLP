"""
models/classification/xlmr_head.py

XLM-R style classification head that operates on pre-computed sentence embeddings.
Architecture mirrors XLM-R's classifier: Dense → Tanh → Dropout → Linear.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _XLMRHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_classes):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.drop(x)
        x = torch.tanh(self.dense(x))
        x = self.drop(x)
        return self.fc(x)


class XLMRClassifierHead:
    def __init__(
        self,
        hidden_dim=768,
        dropout=0.1,
        num_classes=3,
        epochs=20,
        batch_size=64,
        lr=2e-4,
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_fitted = False

    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X, dtype=np.float32)

        self.model = _XLMRHead(
            X.shape[1], self.hidden_dim, self.dropout, self.num_classes
        ).to(self.device)

        X_t = torch.tensor(X).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch + 1}/{self.epochs} — loss: {total_loss / len(loader):.4f}")

        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X, dtype=np.float32)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[i : i + self.batch_size]).to(self.device)
                preds.append(self.model(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)
