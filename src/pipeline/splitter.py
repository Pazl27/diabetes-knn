import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    target_column: str = "Outcome",
    random_state: int = 42,
) -> tuple[Path, Path]:
    # Read the dataset (first row = column headers)
    df = pd.read_csv(input_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. "
                        f"Available columns: {list(df.columns)}")

    # Stratified split to maintain class distribution
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path


def get_split_stats(train_path: Path, test_path: Path, target_column: str = "Outcome") -> dict:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return {
        "train": {
            "total": len(train_df),
            "positive": int(train_df[target_column].sum()),
            "negative": int(len(train_df) - train_df[target_column].sum()),
            "positive_ratio": float(train_df[target_column].mean()),
        },
        "test": {
            "total": len(test_df),
            "positive": int(test_df[target_column].sum()),
            "negative": int(len(test_df) - test_df[target_column].sum()),
            "positive_ratio": float(test_df[target_column].mean()),
        },
    }
