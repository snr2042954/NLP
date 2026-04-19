from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

### Configuration
EMBEDDER_MAP = {
    "TFIDFEmbedder": "TF-IDF",
    "BERTEmbedder": "mBERT",
    "LaBSEEmbedder": "LaBSE",
    "FastTextEmbedder": "FastText",
}

CLASSIFIER_MAP = {
    "LogisticRegressionClassifier": "Logistic Regression",
    "MLPClassifier": "MLP",
    "XLMRClassifierHead": "XLM-R head",
}

LANGUAGE_MAP = {
    "english": "EN",
    "german": "DE",
    "arabic": "AR",
    "portuguese": "PT",
}

EMBEDDER_ORDER = ["TF-IDF", "mBERT", "FastText", "LaBSE"]
CLASSIFIER_ORDER = ["Logistic Regression", "MLP", "XLM-R head"]
LANGUAGE_ORDER = ["EN", "DE", "AR", "PT"]

### Results loading
def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "embedder",
        "classifier",
        "train_language",
        "test_language",
        "accuracy",
        "precision",
        "recall",
        "f1",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["embedder_name"] = df["embedder"].map(EMBEDDER_MAP).fillna(df["embedder"])
    df["classifier_name"] = df["classifier"].map(CLASSIFIER_MAP).fillna(df["classifier"])
    df["train_lang"] = df["train_language"].map(LANGUAGE_MAP).fillna(df["train_language"])
    df["test_lang"] = df["test_language"].map(LANGUAGE_MAP).fillna(df["test_language"])
    return df

### Helpers
def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

def save_figure(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

### Plot 1
def plot_same_language_best(df: pd.DataFrame, out_dir: str) -> None:
    same = df[df["train_language"] == df["test_language"]]
    summary = (
        same.groupby(["embedder_name", "test_lang"], as_index=False)["accuracy"]
        .max()
        .rename(columns={"accuracy": "best_accuracy"})
    )
    summary["embedder_name"] = pd.Categorical(summary["embedder_name"], EMBEDDER_ORDER, ordered=True)
    summary["test_lang"] = pd.Categorical(summary["test_lang"], LANGUAGE_ORDER, ordered=True)
    summary = summary.sort_values(["test_lang", "embedder_name"])

    pivot = summary.pivot(index="embedder_name", columns="test_lang", values="best_accuracy").reindex(EMBEDDER_ORDER)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(EMBEDDER_ORDER))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for offset, lang in zip(offsets, LANGUAGE_ORDER):
        vals = pivot[lang].tolist()
        ax.bar([i + offset for i in x], vals, width=width, label=lang)

    ax.axhline(1 / 3, linestyle="--", linewidth=1)
    ax.text(len(EMBEDDER_ORDER) - 0.25, 1 / 3 + 0.005, "chance", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(EMBEDDER_ORDER)
    ax.set_ylim(0.25, 0.72)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best same-language accuracy by embedder")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    style_axes(ax)
    save_figure(fig, out_dir + "/fig_same_language_best_accuracy.png")

### Plot 2
def plot_english_zero_shot(df: pd.DataFrame, out_dir: str) -> None:
    subset = df[(df["train_language"] == "english") & (df["test_language"] != "english")]
    summary = (
        subset.groupby(["embedder_name", "test_lang"], as_index=False)["accuracy"]
        .max()
        .rename(columns={"accuracy": "best_accuracy"})
    )
    order = ["DE", "AR", "PT"]
    summary["embedder_name"] = pd.Categorical(summary["embedder_name"], EMBEDDER_ORDER, ordered=True)
    summary["test_lang"] = pd.Categorical(summary["test_lang"], order, ordered=True)
    summary = summary.sort_values(["test_lang", "embedder_name"])

    pivot = summary.pivot(index="embedder_name", columns="test_lang", values="best_accuracy").reindex(EMBEDDER_ORDER)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(EMBEDDER_ORDER))
    width = 0.22
    offsets = [-width, 0, width]

    for offset, lang in zip(offsets, order):
        vals = pivot[lang].tolist()
        ax.bar([i + offset for i in x], vals, width=width, label=lang)

    ax.axhline(1 / 3, linestyle="--", linewidth=1)
    ax.text(len(EMBEDDER_ORDER) - 0.25, 1 / 3 + 0.005, "chance", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(EMBEDDER_ORDER)
    ax.set_ylim(0.25, 0.68)
    ax.set_ylabel("Accuracy")
    ax.set_title("Zero-shot transfer from English to other languages")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    style_axes(ax)
    save_figure(fig, out_dir + "/fig_english_zero_shot_accuracy.png")

### Plot 3
def plot_classifier_effect(df: pd.DataFrame, out_dir: str) -> None:
    rows = []
    for (embedder, classifier), group in df.groupby(["embedder_name", "classifier_name"]):
        rows.append(
            {
                "embedder_name": embedder,
                "classifier_name": classifier,
                "same": group.loc[group["train_language"] == group["test_language"], "accuracy"].mean(),
                "cross": group.loc[group["train_language"] != group["test_language"], "accuracy"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["embedder_name"] = pd.Categorical(summary["embedder_name"], EMBEDDER_ORDER, ordered=True)
    summary["classifier_name"] = pd.Categorical(summary["classifier_name"], CLASSIFIER_ORDER, ordered=True)
    summary = summary.sort_values(["embedder_name", "classifier_name"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, column, title in zip(axes, ["same", "cross"], ["Same-language average", "Cross-lingual average"]):
        x = range(len(EMBEDDER_ORDER))
        width = 0.22
        offsets = [-width, 0, width]

        pivot = summary.pivot(index="embedder_name", columns="classifier_name", values=column).reindex(EMBEDDER_ORDER)
        for offset, clf in zip(offsets, CLASSIFIER_ORDER):
            vals = pivot[clf].tolist()
            ax.bar([i + offset for i in x], vals, width=width, label=clf)

        ax.axhline(1 / 3, linestyle="--", linewidth=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(EMBEDDER_ORDER, rotation=0)
        ax.set_title(title)
        ax.set_ylim(0.25, 0.72)
        style_axes(ax)

    axes[0].set_ylabel("Accuracy")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Embedding choice matters far more than classifier choice", y=1.03)
    save_figure(fig, out_dir + "/fig_classifier_vs_embedder.png")

### Plot 4
def plot_labse_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    subset = df[df["embedder_name"] == "LaBSE"]
    summary = (
        subset.groupby(["train_lang", "test_lang"], as_index=False)["accuracy"]
        .max()
        .rename(columns={"accuracy": "best_accuracy"})
    )
    summary["train_lang"] = pd.Categorical(summary["train_lang"], LANGUAGE_ORDER, ordered=True)
    summary["test_lang"] = pd.Categorical(summary["test_lang"], LANGUAGE_ORDER, ordered=True)
    matrix = summary.pivot(index="train_lang", columns="test_lang", values="best_accuracy").reindex(index=LANGUAGE_ORDER, columns=LANGUAGE_ORDER)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(matrix.values, aspect="equal")
    ax.set_xticks(range(len(LANGUAGE_ORDER)))
    ax.set_yticks(range(len(LANGUAGE_ORDER)))
    ax.set_xticklabels(LANGUAGE_ORDER)
    ax.set_yticklabels(LANGUAGE_ORDER)
    ax.set_xlabel("Test language")
    ax.set_ylabel("Train language")
    ax.set_title("LaBSE best accuracy by train/test pair")

    for i in range(len(LANGUAGE_ORDER)):
        for j in range(len(LANGUAGE_ORDER)):
            val = matrix.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    save_figure(fig, out_dir +"/fig_labse_train_test_heatmap.png")


if __name__ == "__main__":

    CSV_PATH = "results.csv"
    OUTPUT_DIR = "figures"

    df = load_results(CSV_PATH)

    plot_same_language_best(df, OUTPUT_DIR)
    plot_english_zero_shot(df, OUTPUT_DIR)
    plot_classifier_effect(df, OUTPUT_DIR)
    plot_labse_heatmap(df, OUTPUT_DIR)

    print(f"Saved figures to: {OUTPUT_DIR}")

