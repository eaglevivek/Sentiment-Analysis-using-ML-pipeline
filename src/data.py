from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "IMDB Dataset.csv"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}.\n"
            "Download the dataset and place it at data/IMDB Dataset.csv."
        )

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df
