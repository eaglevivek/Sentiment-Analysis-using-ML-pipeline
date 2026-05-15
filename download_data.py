from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset() -> None:
    destination = Path("data")
    destination.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(
        "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "IMDB Dataset.csv",
        path=str(destination),
        force=False,
    )

    print(f"Dataset download requested. Verify `data/IMDB Dataset.csv` exists.")


if __name__ == "__main__":
    download_dataset()
