
import pandas as pd
from pathlib import Path

from ..algorithms.knn_classifier import KNNClassifier


def run_prediction(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    random_state: int = 42,
) -> tuple[KNNClassifier, dict]:
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Create and fit classifier (using train set only for k selection)
    classifier = KNNClassifier(n_neighbors=None, random_state=random_state)
    classifier.fit(train_df, val_df=None)

    # Evaluate on test set
    results = classifier.evaluate(test_df)

    # Save predictions
    predictions = classifier.predict(test_df)
    probabilities = classifier.predict_proba(test_df)[:, 1]  # Probability of diabetes

    test_with_predictions = test_df.copy()
    test_with_predictions["Prediction"] = predictions
    test_with_predictions["Probability"] = probabilities
    test_with_predictions.to_csv(output_path, index=False)

    return classifier, results
