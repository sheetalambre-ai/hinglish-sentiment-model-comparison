import os
import re
import json
import random
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

RANDOM_SEED = 42

TEST_SIZE = 0.15
VAL_SIZE = 0.15

OUTPUT_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df = df.rename(
    columns={
        TEXT_COL: "text",
        LABEL_COL: "label"
    }
)[["text", "label"]]

before = len(df)

df = df.dropna(
    subset=["text", "label"]
)

df = df.drop_duplicates(
    subset=["text", "label"]
).reset_index(drop=True)

print(f"Original rows: {before}")
print(f"Rows after deduplication: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 2. TEXT CLEANING
# ============================================================

DEVANAGARI_RANGE = r"\u0900-\u097F"

URL_RE = re.compile(
    r"http\S+|www\.\S+"
)

MENTION_RE = re.compile(
    r"@\w+"
)

KEEP_CHARS_RE = re.compile(
    rf"[^{DEVANAGARI_RANGE}a-zA-Z0-9\s]"
)

MULTISPACE_RE = re.compile(
    r"\s+"
)


def clean_text(text):

    text = str(text)

    text = URL_RE.sub(" ", text)

    text = MENTION_RE.sub(" ", text)

    text = KEEP_CHARS_RE.sub(" ", text)

    text = text.lower()

    text = MULTISPACE_RE.sub(
        " ",
        text
    ).strip()

    return text


df["clean_text"] = df["text"].apply(clean_text)

df = df[
    df["clean_text"].str.len() > 0
].reset_index(drop=True)


# ============================================================
# 3. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["label_id"] = label_encoder.fit_transform(
    df["label"]
)

num_classes = len(
    label_encoder.classes_
)

print("\nClasses:")
print(list(label_encoder.classes_))


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    df["clean_text"].values,
    df["label_id"].values,

    test_size=TEST_SIZE,

    stratify=df["label_id"].values,

    random_state=RANDOM_SEED
)


val_relative = VAL_SIZE / (1 - TEST_SIZE)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,

    test_size=val_relative,

    stratify=y_temp,

    random_state=RANDOM_SEED
)


print("\nDataset split:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")


# ============================================================
# 5. TOKENIZATION
# ============================================================

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

# IMPORTANT:
# Fit tokenizer ONLY on training data
tokenizer.fit_on_texts(X_train)

vocab_size = min(
    MAX_VOCAB_SIZE,
    len(tokenizer.word_index) + 1
)

print(f"\nVocabulary size: {vocab_size}")


def to_padded(texts):

    sequences = tokenizer.texts_to_sequences(
        texts
    )

    return pad_sequences(
        sequences,
        maxlen=MAX_SEQ_LEN,
        padding="post",
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

class_weights = dict(
    zip(
        np.unique(y_train),
        class_weight_values
    )
)

print("\nClass weights:")
print(class_weights)


# ============================================================
# 7. MODEL ARCHITECTURES
# ============================================================

architectures = {

    "Simple LSTM":
        build_simple_lstm,

    "Stacked BiLSTM":
        build_stacked_bilstm,

    "CNN + BiLSTM":
        build_cnn_bilstm
}


# ============================================================
# 8. TRAIN + EVALUATE
# ============================================================

results = []

trained_models = {}

histories = {}

all_predictions = {}


for architecture_name, builder in architectures.items():

    print("\n")
    print("=" * 70)
    print(f"TRAINING: {architecture_name}")
    print("=" * 70)

    model = builder(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes
    )

    model.summary()


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = model.fit(

        X_train_pad,
        y_train,

        validation_data=(
            X_val_pad,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=[
            early_stopping
        ],

        verbose=2
    )


    trained_models[
        architecture_name
    ] = model

    histories[
        architecture_name
    ] = history


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_probability = model.predict(
        X_test_pad,
        verbose=0
    )

    y_pred = np.argmax(
        y_probability,
        axis=1
    )

    all_predictions[
        architecture_name
    ] = y_pred


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # RMSE on label-index space.
    # This is only an ordinal proxy.
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )


    print("\nResults:")
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro-F1: {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )


    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({

        "Architecture":
            architecture_name,

        "Accuracy":
            round(accuracy, 4),

        "Macro_F1":
            round(macro_f1, 4),

        "Weighted_F1":
            round(weighted_f1, 4),

        "RMSE":
            round(rmse, 4),

        "Params":
            model.count_params(),

        "Epochs_Run":
            len(
                history.history["loss"]
            )
    })


# ============================================================
# 9. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    "Macro_F1",
    ascending=False
)


print("\n")
print("=" * 70)
print("FINAL ARCHITECTURE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "architecture_comparison.csv"
    ),
    index=False
)


# ============================================================
# 10. MODEL COMPARISON CHART
# ============================================================

metrics = [
    "Accuracy",
    "Macro_F1",
    "RMSE"
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)


for ax, metric in zip(
    axes,
    metrics
):

    ax.bar(
        results_df["Architecture"],
        results_df[metric]
    )

    ax.set_title(metric)

    ax.set_ylabel(metric)

    ax.tick_params(
        axis="x",
        rotation=20
    )

    for i, value in enumerate(
        results_df[metric]
    ):

        ax.text(
            i,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "architecture_comparison.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

best_architecture = results_df.iloc[0][
    "Architecture"
]

best_predictions = all_predictions[
    best_architecture
]

cm = confusion_matrix(
    y_test,
    best_predictions
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)

im = ax.imshow(cm)

ax.set_title(
    f"Confusion Matrix - {best_architecture}"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "True Label"
)

ax.set_xticks(
    range(num_classes)
)

ax.set_yticks(
    range(num_classes)
)

ax.set_xticklabels(
    label_encoder.classes_
)

ax.set_yticklabels(
    label_encoder.classes_
)


for i in range(num_classes):

    for j in range(num_classes):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 12. TRAINING HISTORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


for architecture_name, history in histories.items():

    ax.plot(
        history.history["val_accuracy"],
        label=architecture_name
    )


ax.set_title(
    "Validation Accuracy"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Accuracy"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "training_history.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

best_model = trained_models[
    best_architecture
]

best_model.save(
    os.path.join(
        OUTPUT_DIR,
        "best_model.keras"
    )
)


# ============================================================
# 14. SAVE TOKENIZER
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "tokenizer.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        tokenizer.to_json()
    )


# ============================================================
# 15. SAVE LABEL ENCODER
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "label_classes.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        list(label_encoder.classes_),
        file,
        ensure_ascii=False,
        indent=4
    )


print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best architecture: {best_architecture}"
)

print(
    "Saved model: results/best_model.keras"
)