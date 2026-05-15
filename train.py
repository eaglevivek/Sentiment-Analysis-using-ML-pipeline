from pathlib import Path
from src.pipeline import train_and_evaluate, save_model

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"


def main() -> None:
    model, accuracy, report = train_and_evaluate()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Classification report:\n")
    print(report)


if __name__ == "__main__":
    main()
