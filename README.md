# IMDB Sentiment Analyzer

A simple machine learning pipeline for sentiment classification of IMDB movie reviews, with a Streamlit web app for inference.

## Project overview

- `train.py` trains a text classification pipeline using `TfidfVectorizer` + `LogisticRegression`.
- `app.py` is a Streamlit app that loads the trained model and predicts sentiment from user-provided reviews.
- `src/` contains dataset loading and training pipeline utilities.
- `download_data.py` can download the IMDB dataset from Kaggle when `kaggle.json` is configured.

## Dataset

This project uses the Kaggle dataset:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place the CSV file at:

```bash
data/IMDB Dataset.csv
```

or use the helper script if Kaggle credentials are available.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset manually or with Kaggle:

```bash
python download_data.py
```

4. Train the model:

```bash
python train.py
```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app and select this repository.
3. Set the main file to `app.py`.
4. Add `data/IMDB Dataset.csv` to the repo if you want the app to work without running training first, or include a trained model in `models/sentiment_model.joblib`.

> For best results, train locally first and commit the generated model to the repository if you want the deployed app to load immediately.
