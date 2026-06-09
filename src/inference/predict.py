import sys
import os
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.append(BASE_DIR)

from src.data.preprocess import clean_text


# ==========================
# PATHS
# ==========================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "toxicity_model_final.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "tokenizer.pkl"
)


# ==========================
# LOAD MODEL & TOKENIZER
# ==========================

model = None
tokenizer = None


def load_artifacts():
    global model, tokenizer

    if model is None:
        model = load_model(MODEL_PATH)

    if tokenizer is None:
        with open(TOKENIZER_PATH, "rb") as file:
            tokenizer = pickle.load(file)


# ==========================
# CONFIG
# ==========================

TARGET_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

MAX_LEN = 200


# ==========================
# PREDICTION
# ==========================

def predict_comment(comment):

    load_artifacts()

    cleaned_text = clean_text(comment)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    results = {}

    for label, score in zip(
        TARGET_COLUMNS,
        prediction
    ):
        results[label] = float(score)

    PREDICTION_THRESHOLD = 0.3

    toxic_labels = [
        label
        for label, score in results.items()
        if score >= PREDICTION_THRESHOLD
    ]
    prediction_class = (
        "Toxic"
        if len(toxic_labels) > 0
        else "Non Toxic"
    )

    return {
        "prediction": prediction_class,
        "detected_labels": toxic_labels,
        "scores": results
    }


# ==========================
# TEST
# ==========================

if __name__ == "__main__":

    result = predict_comment(
        "You are a stupid idiot"
    )

    print(result)