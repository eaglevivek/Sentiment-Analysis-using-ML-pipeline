import random
from pathlib import Path

import pandas as pd
import joblib
import streamlit as st

from src.data import load_data

MODEL_PATH = Path(__file__).resolve().parent / "models" / "sentiment_model.joblib"
DATA_PATH = Path(__file__).resolve().parent / "data" / "IMDB Dataset.csv"

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` first to generate models/sentiment_model.joblib."
        )
    return joblib.load(path)

@st.cache_data
def load_sample_reviews(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df["review"].dropna().astype(str).tolist()

@st.cache_data
def random_review_sample() -> str:
    reviews = load_sample_reviews(DATA_PATH)
    if reviews:
        return random.choice(reviews)

    fallback = [
        "An unforgettable emotional rollercoaster with strong performances and a beautiful score.",
        "A predictable plot with wooden dialogue and an uninteresting hero.",
        "The cinematography is breathtaking, but the pacing feels slow at times.",
        "This movie is a fun, light-hearted comedy that keeps the audience laughing.",
        "A dark and powerful drama that leaves a lasting impression.",
    ]
    return random.choice(fallback)

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")
st.title("IMDB Sentiment Analyzer")
st.write(
    "Enter a movie review and the model will predict whether it is positive or negative."
)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as error:
    st.error(error)
    st.info("Train the model with `python train.py` or upload a trained model to `models/sentiment_model.joblib`.")
    st.stop()

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Load random review"):
        st.session_state.review_text = random_review_sample()
with col2:
    if st.button("Clear review"):
        st.session_state.review_text = ""

review = st.text_area("Movie review", value=st.session_state.review_text, height=200)
st.session_state.review_text = review

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([review.strip()])[0]
        label = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted sentiment: {label}")
        if prediction == 1:
            st.write("The model believes this review has a positive tone.")
        else:
            st.write("The model believes this review has a negative tone.")
