import streamlit as st
import joblib
from pathlib import Path

MODEL_PATH = Path("models") / "sentiment_model.joblib"

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` first to generate models/sentiment_model.joblib."
        )
    return joblib.load(path)

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")
st.title("IMDB Sentiment Sentiment Analyzer")
st.write(
    "Enter a movie review and the model will predict whether it is positive or negative."
)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as error:
    st.error(error)
    st.info("Train the model with `python train.py` or upload a trained model to `models/sentiment_model.joblib`.")
    st.stop()

review = st.text_area("Movie review", height=200)

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
