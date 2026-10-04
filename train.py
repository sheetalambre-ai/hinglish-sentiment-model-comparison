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

from preprocessing import clean_text, dedup_key, has_stock_phrase

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
# 2. TEXT CLEANING + NEAR-DUPLICATE REMOVAL
# ============================================================

df["clean_text"] = df["text"].apply(clean_text)

df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

# Near-duplicate key: stock phrases removed, words sorted (see
# preprocessing.dedup_key). Augmented copies of the same review
# ("to be honest, <review>", shuffled sentences) share one key.
df["dedup_key"] = df["clean_text"].apply(dedup_key)
df["has_stock_phrase"] = df["clean_text"].apply(has_stock_phrase)

# Kept for the leakage audit in section 14
df_exact_dedup = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

# Reviews sharing a key but carrying different labels are ambiguous,
# so they are dropped entirely.
labels_per_key = df.groupby("dedup_key")["label"].nunique()
conflicting = labels_per_key[labels_per_key > 1].index

df = df[~df["dedup_key"].isin(conflicting)]

# Keep ONE review per key, so neither an exact copy nor an augmented
# copy of a review can appear in both the training and the test set.
df = df.drop_duplicates(subset=["dedup_key"]).reset_index(drop=True)

print(f"Reviews containing a stock phrase: {df_exact_dedup['has_stock_phrase'].sum()}")
print(f"Rows after removing exact duplicates only: {len(df_exact_dedup)}")
print(f"Conflicting-label keys removed: {len(conflicting)}")
print(f"Rows after removing near-duplicates: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 3. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["label_id"] = label_encoder.fit_transform(df["label"])
df_exact_dedup["label_id"] = label_encoder.transform(df_exact_dedup["label"])

num_classes = len(label_encoder.classes_)

print("\nClasses:")
print(list(label_encoder.classes_))


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT (fixed)
# ============================================================

def make_split(frame):
    """Fixed stratified 70 / 15 / 15 split; returns index arrays."""
    idx = np.arange(len(frame))
    labels = frame["label_id"].values

    idx_temp, idx_test = train_test_split(
        idx,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=SPLIT_SEED
    )

    val_relative = VAL_SIZE / (1 - TEST_SIZE)

    idx_train, idx_val = train_test_split(
        idx_temp,
        test_size=val_relative,
        stratify=labels[idx_temp],
        random_state=SPLIT_SEED
    )

    return idx_train, idx_val, idx_test


idx_train, idx_val, idx_test = make_split(df)

X_train, y_train = df["clean_text"].values[idx_train], df["label_id"].values[idx_train]
X_val, y_val = df["clean_text"].values[idx_val], df["label_id"].values[idx_val]
X_test, y_test = df["clean_text"].values[idx_test], df["label_id"].values[idx_test]

# Sanity check: no review (or near-duplicate of one) is shared
# between the splits
keys = df["dedup_key"].values
assert not set(keys[idx_train]) & set(keys[idx_test])
assert not set(keys[idx_train]) & set(keys[idx_val])
assert not set(keys[idx_val]) & set(keys[idx_test])

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
def make_padder(tok):
    def pad(texts):
        sequences = tok.texts_to_sequences(texts)
        return pad_sequences(
            sequences,
            maxlen=MAX_SEQ_LEN,
            padding="pre",
            truncating="post"
        )
    return pad


to_padded = make_padder(tokenizer)


X_train_pad = to_padded(X_train)
X_val_pad = to_padded(X_val)
X_test_pad = to_padded(X_test)


# ============================================================
# 6. CLASS WEIGHTS
# ============================================================

def balanced_weights(y):
    classes = np.unique(y)
    values = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, values))


class_weights = balanced_weights(y_train)

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

def train_architectures(Xtr, ytr, Xva, yva, Xte, yte, vocab, cw,
                        show_summary=True, tag=""):
    """Train every architecture with every seed; evaluate on (Xte, yte)."""

    run_results = []
    runs = {}   # (architecture, seed) -> dict(model, history, y_pred)

    for architecture_name, builder in architectures.items():

        for seed in TRAINING_SEEDS:

            print("\n")
            print("=" * 70)
            print(f"TRAINING{tag}: {architecture_name} | seed {seed}")
            print("=" * 70)

            # Seeds Python, NumPy and TensorFlow in one call
            tf.keras.utils.set_random_seed(seed)

            model = builder(
                vocab_size=vocab,
                max_seq_len=MAX_SEQ_LEN,
                embedding_dim=EMBEDDING_DIM,
                num_classes=num_classes
            )

            if show_summary and seed == TRAINING_SEEDS[0]:
                model.summary()

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True
            )

            history = model.fit(
                Xtr,
                ytr,
                validation_data=(Xva, yva),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                class_weight=cw,
                callbacks=[early_stopping],
                verbose=2
            )

            y_pred = np.argmax(model.predict(Xte, verbose=0), axis=1)

            accuracy = accuracy_score(yte, y_pred)
            macro_f1 = f1_score(yte, y_pred, average="macro")
            weighted_f1 = f1_score(yte, y_pred, average="weighted")

            # RMSE on label-index space (negative=0, neutral=1, positive=2).
            # This is only an ordinal proxy.
            rmse = np.sqrt(mean_squared_error(yte, y_pred))

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

    return run_results, runs


run_results, runs = train_architectures(
    X_train_pad, y_train,
    X_val_pad, y_val,
    X_test_pad, y_test,
    vocab_size, class_weights
)


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


# ============================================================
# 14. LEAKAGE AUDIT: EXACT vs NEAR-DUPLICATE REMOVAL
# ============================================================

# Re-runs the experiment the OLD way (only exact duplicates removed)
# to measure how much the augmented copies inflate test scores.
# Test reviews are split into "leaked" (a near-duplicate is in the
# training or validation set) and "clean" (no near-duplicate).

RUN_LEAKAGE_AUDIT = True

if RUN_LEAKAGE_AUDIT:

    leaky = df_exact_dedup
    l_train, l_val, l_test = make_split(leaky)

    l_text = leaky["clean_text"].values
    l_y = leaky["label_id"].values
    l_keys = leaky["dedup_key"].values

    seen_keys = set(l_keys[l_train]) | set(l_keys[l_val])
    leaked_mask = np.array([k in seen_keys for k in l_keys[l_test]])

    print("\n")
    print("=" * 70)
    print("LEAKAGE AUDIT (exact-duplicate removal only)")
    print("=" * 70)
    print(f"Rows: {len(leaky)} | test reviews: {len(l_test)}")
    print(
        f"Test reviews with a near-duplicate in train/val: "
        f"{leaked_mask.sum()} ({leaked_mask.mean():.1%})"
    )

    l_tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    l_tokenizer.fit_on_texts(l_text[l_train])
    l_pad = make_padder(l_tokenizer)
    l_vocab = min(MAX_VOCAB_SIZE, len(l_tokenizer.word_index) + 1)

    _, leaky_runs = train_architectures(
        l_pad(l_text[l_train]), l_y[l_train],
        l_pad(l_text[l_val]), l_y[l_val],
        l_pad(l_text[l_test]), l_y[l_test],
        l_vocab, balanced_weights(l_y[l_train]),
        show_summary=False, tag=" [leakage audit]"
    )

    audit_rows = []
    y_l_test = l_y[l_test]

    for (arch, seed), run in leaky_runs.items():
        pred = run["y_pred"]
        row = {"Architecture": arch, "Seed": seed}
        for name, mask in [
            ("All", np.ones_like(leaked_mask)),
            ("Leaked", leaked_mask),
            ("Clean", ~leaked_mask),
        ]:
            row[f"Acc_{name}"] = accuracy_score(y_l_test[mask], pred[mask])
            row[f"MacroF1_{name}"] = f1_score(y_l_test[mask], pred[mask], average="macro")
        audit_rows.append(row)

    audit_runs = pd.DataFrame(audit_rows)
    audit_runs.round(4).to_csv(
        os.path.join(OUTPUT_DIR, "leakage_audit_runs.csv"), index=False
    )

    audit_cols = [c for c in audit_runs.columns if c.startswith(("Acc_", "MacroF1_"))]
    audit_summary = audit_runs.groupby("Architecture", sort=False)[audit_cols].agg(["mean", "std"])
    audit_summary.columns = [f"{c}_{s}" for c, s in audit_summary.columns]
    audit_summary = audit_summary.reset_index()

    # Clean-protocol (near-duplicate removal) results for comparison
    clean_acc = summary.set_index("Architecture")[["Accuracy_mean", "Accuracy_std"]]
    audit_summary["Acc_NearDedup_mean"] = audit_summary["Architecture"].map(clean_acc["Accuracy_mean"])
    audit_summary["Acc_NearDedup_std"] = audit_summary["Architecture"].map(clean_acc["Accuracy_std"])

    audit_summary.round(4).to_csv(
        os.path.join(OUTPUT_DIR, "leakage_audit.csv"), index=False
    )

    pretty = audit_summary[["Architecture"]].copy()
    for name, col in [
        ("Exact-dedup: all test", "Acc_All"),
        ("Exact-dedup: leaked", "Acc_Leaked"),
        ("Exact-dedup: clean", "Acc_Clean"),
        ("Near-dedup (main result)", "Acc_NearDedup"),
    ]:
        pretty[name] = (
            audit_summary[f"{col}_mean"].map("{:.4f}".format)
            + " ± "
            + audit_summary[f"{col}_std"].map("{:.4f}".format)
        )
    print("\nTest accuracy (mean ± std over seeds):")
    print(pretty.to_string(index=False))

    # Chart
    fig, ax = plt.subplots(figsize=(11, 5))
    groups = [
        ("Exact-dedup: leaked test reviews", "Acc_Leaked"),
        ("Exact-dedup: clean test reviews", "Acc_Clean"),
        ("Near-dedup (main result)", "Acc_NearDedup"),
    ]
    width = 0.26
    x = np.arange(len(audit_summary))
    for i, (label, col) in enumerate(groups):
        ax.bar(
            x + (i - 1) * width,
            audit_summary[f"{col}_mean"],
            width,
            yerr=audit_summary[f"{col}_std"],
            capsize=4,
            label=label
        )
    ax.set_xticks(x)
    ax.set_xticklabels(audit_summary["Architecture"])
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Leakage audit (mean ± std, {len(TRAINING_SEEDS)} seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "leakage_audit.png"), dpi=150)
    plt.close()


print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Best architecture: {best_architecture} (seed {best_seed})")
print("Saved model: results/best_model.keras")
