import json

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import clean_text


MODEL_PATH = "./results/best_model.keras"
TOKENIZER_PATH = "./results/tokenizer.json"
LABEL_PATH = "./results/label_classes.json"

MAX_SEQ_LEN = 60


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

with open(
    TOKENIZER_PATH,
    "r",
    encoding="utf-8"
) as file:

    tokenizer = tokenizer_from_json(
        file.read()
    )


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    labels = json.load(file)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):

    sequence = tokenizer.texts_to_sequences(
        [clean_text(text)]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_SEQ_LEN,
        padding="pre",
        truncating="post"
    )

    probabilities = model.predict(
        padded,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = labels[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return predicted_label, confidence


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HINDI / HINGLISH SENTIMENT ANALYZER")
    print("=" * 60)

    while True:

        text = input(
            "\nEnter a review "
            "(or type 'exit'): "
        )

        if text.lower() == "exit":
            break

        sentiment, confidence = predict_sentiment(
            text
        )

        print(
            f"Sentiment: {sentiment}"
        )

        print(
            f"Confidence: {confidence:.2%}"
        )