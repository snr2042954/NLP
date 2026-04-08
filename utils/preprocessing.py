"""
utils/preprocessing.py
"""

import re
import emoji


LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_MAP = {v: k for k, v in LABEL_MAP.items()}


def preprocess(text: str) -> str:
    text = str(text)

    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"https?://\S+|http\S+|www\S+|\bhttps?\b|\bhtt\b", "", text)
    text = re.sub(r"\bRT\b", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_preprocessing(df):
    """
    Apply text cleaning + label encoding.
    Returns a new dataframe.
    """
    df = df.copy()

    # Clean text
    df["text"] = df["text"].apply(preprocess)

    # Encode labels
    df["label"] = df["label"].map(LABEL_MAP)

    return df


def decode_labels(labels):
    return [ID_MAP[int(l)] for l in labels]