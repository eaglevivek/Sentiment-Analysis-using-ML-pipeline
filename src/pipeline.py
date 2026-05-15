from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_data


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_model.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_df=0.95,
                    min_df=5,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_evaluate(test_size: float = 0.2, random_state: int = 42) -> tuple[Pipeline, float, str]:
    df = load_data()
    X = df["review"].astype(str)
    y = df["sentiment"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["negative", "positive"])

    return pipeline, accuracy, report


def save_model(model: Pipeline, target_path: str | Path = MODEL_PATH) -> None:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
