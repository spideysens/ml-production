"""Train and export the Amazon review sentiment model."""

import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "amazon_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


def load_dataset() -> tuple[list[str], list[int]]:
    """Load non-empty reviews and binary labels from the workshop dataset."""
    reviews: list[str] = []
    labels: list[int] = []

    with DATASET_PATH.open("r", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)

        for row in reader:
            review = row["reviewText"].strip()
            label = row["Positive"].strip()

            if review and label in {"0", "1"}:
                reviews.append(review)
                labels.append(int(label))

    if not reviews:
        raise ValueError(f"No valid rows found in {DATASET_PATH}")

    return reviews, labels


def main() -> None:
    reviews, labels = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000)),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, digits=4))
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
