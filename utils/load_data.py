"""
utils/load_data.py

Handles loading the multilingual sentiment dataset.

Returns a pandas DataFrame with columns:
- text
- label
- language
"""

import os
import pandas as pd


def load_data(
    languages=("english", "german", "chinese"),
    split="train",
    frac=1.0,
    random_state=42,
):
    """
    Load dataset from raw CSV files.

    Parameters
    ----------
    languages : tuple
        Languages to include
    split : str
        "train", "test", or "valid"
    frac : float
        Fraction of data to sample
    random_state : int
        Seed for reproducibility

    Returns
    -------
    pd.DataFrame
    """

    base_url = "https://raw.githubusercontent.com/tyqiangz/multilingual-sentiment-datasets/main/data"

    dfs = []

    for lang in languages:
        url = f"{base_url}/{lang}/{split}.csv"

        try:
            df = pd.read_csv(url)
        except Exception as e:
            raise RuntimeError(f"Failed to load {url}: {e}")

        # Keep only required columns
        df = df[["text", "label"]].copy()
        df["language"] = lang

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    if frac < 1.0:
        data = data.sample(frac=frac, random_state=random_state)

    return data.reset_index(drop=True)