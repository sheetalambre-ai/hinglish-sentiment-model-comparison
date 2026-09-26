import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam


def build_simple_lstm(
    vocab_size,
    max_seq_len,
    embedding_dim,
    num_classes
):
    """
    Simple LSTM baseline model.
    """

    model = Sequential([
        Input(shape=(max_seq_len,)),

        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        ),

        SpatialDropout1D(0.3),

        LSTM(
            64,
            dropout=0.3,
            recurrent_dropout=0.2
        ),

        Dense(32, activation="relu"),

        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        )
    ], name="A_Simple_LSTM")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model