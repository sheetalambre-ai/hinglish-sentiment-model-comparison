import os
import json
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import clean_text

from models.simple_lstm import build_simple_lstm
from models.stacked_bilstm import build_stacked_bilstm
from models.cnn_bilstm import build_cnn_bilstm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "./data/hinglish_reviews.csv"

TEXT_COL = "Review"
LABEL_COL = "Sentiment"

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 60
EMBEDDING_DIM = 128

BATCH_SIZE = 32
EPOCHS = 25

# The data split is fixed with SPLIT_SEED so every run is tested on
# the same reviews. Each architecture is trained once per seed in
# TRAINING_SEEDS (weight initialisation, dropout, batch shuffling).
SPLIT_SEED = 42
TRAINING_SEEDS = [42, 123, 2024]

TEST_SIZE = 0.15
VAL_SIZE = 0.15

OUTPUT_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df = df.rename(
    columns={TEXT_COL: "text", LABEL_COL: "label"}
)[["text", "label"]]

df = df.dropna(subset=["text", "label"])

print(f"Original rows: {len(df)}")


# ============================================================
# 2. TEXT CLEANING + DEDUPLICATION (after cleaning)
# ============================================================

df["clean_text"] = df["text"].apply(clean_text)

df = df[df["clean_text"].str.len() > 0]

# Reviews that become identical after cleaning but carry different
# labels are ambiguous, so they are dropped entirely.
labels_per_text = df.groupby("clean_text")["label"].nunique()
conflicting = labels_per_text[labels_per_text > 1].index

df = df[~df["clean_text"].isin(conflicting)]

# Remove duplicates on the CLEANED text, so the same review cannot
# appear in both the training and the test set.
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

print(f"Conflicting-label texts removed: {len(conflicting)}")
print(f"Rows after cleaning + deduplication: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 3. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["label_id"] = label_encoder.fit_transform(df["label"])

num_classes = len(label_encoder.classes_)

print("\nClasses:")
print(list(label_encoder.classes_))


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT (fixed)
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    df["clean_text"].values,
    df["label_id"].values,
    test_size=TEST_SIZE,
    stratify=df["label_id"].values,
    random_state=SPLIT_SEED
)

val_relative = VAL_SIZE / (1 - TEST_SIZE)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=val_relative,
    stratify=y_temp,
    random_state=SPLIT_SEED
)

# Sanity check: no review is shared between the splits
assert not set(X_train) & set(X_test)
assert not set(X_train) & set(X_val)
assert not set(X_val) & set(X_test)

print("\nDataset split:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")


# ============================================================
# 5. TOKENIZATION
# ============================================================

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")

# Fit tokenizer ONLY on training data
tokenizer.fit_on_texts(X_train)

vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)

print(f"\nVocabulary size: {vocab_size}")


# Pre-padding: zeros go BEFORE the review, so the LSTM reads the real
# words last and its final hidden state is not diluted by ~40 padding
# steps (the median review is 18 tokens). Post-padding made the
# one-directional Simple LSTM fail to train on 2 of 3 seeds.
# Reviews longer than MAX_SEQ_LEN (~3%) keep their first 60 tokens.
def to_padded(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding="pre",
        truncating="post"
    )


X_train_pad = to_padded(X_train)
X_val_pad = to_padded(X_val)
X_test_pad = to_padded(X_test)


# ============================================================
# 6. CLASS WEIGHTS
# ============================================================

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(zip(np.unique(y_train), class_weight_values))

print("\nClass weights:")
print(class_weights)


# ============================================================
# 7. MODEL ARCHITECTURES
# ============================================================

architectures = {
    "Simple LSTM": build_simple_lstm,
    "Stacked BiLSTM": build_stacked_bilstm,
    "CNN + BiLSTM": build_cnn_bilstm
}


# ============================================================
# 8. TRAIN + EVALUATE (each architecture x each seed)
# ============================================================

run_results = []
runs = {}   # (architecture, seed) -> dict(model, history, y_pred)

for architecture_name, builder in architectures.items():

    for seed in TRAINING_SEEDS:

        print("\n")
        print("=" * 70)
        print(f"TRAINING: {architecture_name} | seed {seed}")
        print("=" * 70)

        # Seeds Python, NumPy and TensorFlow in one call
        tf.keras.utils.set_random_seed(seed)

        model = builder(
            vocab_size=vocab_size,
            max_seq_len=MAX_SEQ_LEN,
            embedding_dim=EMBEDDING_DIM,
            num_classes=num_classes
        )

        if seed == TRAINING_SEEDS[0]:
            model.summary()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_pad,
            y_train,
            validation_data=(X_val_pad, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=[early_stopping],
            verbose=2
        )

        y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        # RMSE on label-index space (negative=0, neutral=1, positive=2).
        # This is only an ordinal proxy.
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(
            f"\nAccuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f} | "
            f"Weighted-F1: {weighted_f1:.4f} | RMSE: {rmse:.4f}"
        )

        run_results.append({
            "Architecture": architecture_name,
            "Seed": seed,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "RMSE": rmse,
            "Params": model.count_params(),
            "Epochs_Run": len(history.history["loss"])
        })

        runs[(architecture_name, seed)] = {
            "model": model,
            "history": history.history,
            "y_pred": y_pred
        }


# ============================================================
# 9. COMPARISON TABLES (per run + mean ± std)
# ============================================================

runs_df = pd.DataFrame(run_results)

runs_df.round(4).to_csv(
    os.path.join(OUTPUT_DIR, "architecture_runs.csv"),
    index=False
)

metric_cols = ["Accuracy", "Macro_F1", "Weighted_F1", "RMSE"]

summary = runs_df.groupby("Architecture", sort=False).agg(
    **{f"{m}_mean": (m, "mean") for m in metric_cols},
    **{f"{m}_std": (m, "std") for m in metric_cols},
    Params=("Params", "first"),
    Epochs_Mean=("Epochs_Run", "mean")
).reset_index()

summary = summary[
    ["Architecture"]
    + [c for m in metric_cols for c in (f"{m}_mean", f"{m}_std")]
    + ["Params", "Epochs_Mean"]
].sort_values("Macro_F1_mean", ascending=False)

summary.round(4).to_csv(
    os.path.join(OUTPUT_DIR, "architecture_comparison.csv"),
    index=False
)

print("\n")
print("=" * 70)
print(f"FINAL ARCHITECTURE COMPARISON (mean ± std over {len(TRAINING_SEEDS)} seeds)")
print("=" * 70)

pretty = summary[["Architecture"]].copy()
for m in metric_cols:
    pretty[m] = (
        summary[f"{m}_mean"].map("{:.4f}".format)
        + " ± "
        + summary[f"{m}_std"].map("{:.4f}".format)
    )
pretty["Params"] = summary["Params"]
print(pretty.to_string(index=False))


# ============================================================
# 10. MODEL COMPARISON CHART (with std error bars)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric in zip(axes, ["Accuracy", "Macro_F1", "RMSE"]):

    means = summary[f"{metric}_mean"].values
    stds = summary[f"{metric}_std"].values

    ax.bar(summary["Architecture"], means, yerr=stds, capsize=6)
    ax.set_title(f"{metric} (mean ± std, {len(TRAINING_SEEDS)} seeds)")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std, f"{mean:.3f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "architecture_comparison.png"), dpi=150)
plt.close()


# ============================================================
# 11. BEST MODEL + CONFUSION MATRIX
# ============================================================

# Best architecture = highest MEAN Macro-F1; within it, the seed with
# the highest Macro-F1 is kept as the deployable model.
best_architecture = summary.iloc[0]["Architecture"]

best_seed = int(
    runs_df[runs_df["Architecture"] == best_architecture]
    .sort_values("Macro_F1", ascending=False)
    .iloc[0]["Seed"]
)

best_run = runs[(best_architecture, best_seed)]

print("\nClassification report - "
      f"{best_architecture} (seed {best_seed}):")
print(classification_report(
    y_test,
    best_run["y_pred"],
    target_names=label_encoder.classes_
))

cm = confusion_matrix(y_test, best_run["y_pred"])

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Blues")
ax.set_title(f"Confusion Matrix - {best_architecture} (seed {best_seed})")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(label_encoder.classes_)
ax.set_yticklabels(label_encoder.classes_)

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()


# ============================================================
# 12. TRAINING HISTORY (train vs validation loss + accuracy)
# ============================================================

# Plotted for the first seed of each architecture. A widening gap
# between training and validation loss indicates overfitting.
history_seed = TRAINING_SEEDS[0]

fig, axes = plt.subplots(2, len(architectures), figsize=(16, 9))

for col, architecture_name in enumerate(architectures):

    h = runs[(architecture_name, history_seed)]["history"]
    epochs = range(1, len(h["loss"]) + 1)

    ax = axes[0, col]
    ax.plot(epochs, h["loss"], marker="o", label="Train loss")
    ax.plot(epochs, h["val_loss"], marker="o", label="Validation loss")
    ax.set_title(f"{architecture_name} - Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1, col]
    ax.plot(epochs, h["accuracy"], marker="o", label="Train accuracy")
    ax.plot(epochs, h["val_accuracy"], marker="o", label="Validation accuracy")
    ax.set_title(f"{architecture_name} - Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

fig.suptitle(f"Training history (seed {history_seed})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150)
plt.close()


# ============================================================
# 13. SAVE BEST MODEL, TOKENIZER, LABELS
# ============================================================

best_run["model"].save(os.path.join(OUTPUT_DIR, "best_model.keras"))

with open(os.path.join(OUTPUT_DIR, "tokenizer.json"), "w", encoding="utf-8") as file:
    file.write(tokenizer.to_json())

with open(os.path.join(OUTPUT_DIR, "label_classes.json"), "w", encoding="utf-8") as file:
    json.dump(list(label_encoder.classes_), file, ensure_ascii=False, indent=4)


print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Best architecture: {best_architecture} (seed {best_seed})")
print("Saved model: results/best_model.keras")
