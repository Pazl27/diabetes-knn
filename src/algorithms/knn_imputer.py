import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class KNNImputer:

    # Columns where 0 is not a valid value and should be imputed
    COLUMNS_TO_IMPUTE = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    def __init__(
        self,
        n_neighbors: int = 5,
        random_state: int = 42,
        class_aware: bool = True,
        weights: str = "distance",
    ):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.class_aware = class_aware
        self.weights = weights
        self.scalers = {}
        self.global_medians = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "KNNImputer":
        data = df.copy()
        cols_to_impute = [c for c in self.COLUMNS_TO_IMPUTE if c in data.columns]

        # Calculate global medians as fallback
        for col in cols_to_impute:
            non_zero_values = data[data[col] != 0][col]
            self.global_medians[col] = (
                non_zero_values.median() if len(non_zero_values) > 0 else data[col].median()
            )

        # Fit scalers
        feature_cols = [c for c in data.columns if c != "Outcome"]

        if self.class_aware and "Outcome" in data.columns:
            for class_label in data["Outcome"].unique():
                class_data = data[data["Outcome"] == class_label][feature_cols].copy()

                # Replace zeros with medians temporarily for scaler fitting
                for col in cols_to_impute:
                    class_data[col] = class_data[col].replace(0, np.nan)
                    col_median = class_data[col].dropna().median()
                    if pd.isna(col_median):
                        col_median = self.global_medians.get(col, 0)
                    class_data[col] = class_data[col].fillna(col_median)

                scaler = StandardScaler()
                scaler.fit(class_data)
                self.scalers[class_label] = scaler
        else:
            class_data = data[feature_cols].copy()
            for col in cols_to_impute:
                class_data[col] = class_data[col].replace(0, np.nan)
                col_median = class_data[col].dropna().median()
                if pd.isna(col_median):
                    col_median = self.global_medians.get(col, 0)
                class_data[col] = class_data[col].fillna(col_median)

            scaler = StandardScaler()
            scaler.fit(class_data)
            self.scalers[None] = scaler

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Imputer not fitted. Call fit() first.")

        data = df.copy()
        cols_to_impute = [c for c in self.COLUMNS_TO_IMPUTE if c in data.columns]
        feature_cols = [c for c in data.columns if c != "Outcome"]

        # Convert to float
        for col in cols_to_impute:
            data[col] = data[col].astype(float)

        if self.class_aware and "Outcome" in data.columns:
            # Impute within each class
            for class_label in data["Outcome"].unique():
                class_mask = data["Outcome"] == class_label
                class_indices = data[class_mask].index

                if class_label not in self.scalers:
                    # Unknown class - use global medians
                    for col in cols_to_impute:
                        zero_mask = (data[col] == 0) & class_mask
                        data.loc[zero_mask, col] = self.global_medians[col]
                    continue

                class_data = data.loc[class_indices].copy()
                imputed_class_data = self._impute_class(
                    class_data, cols_to_impute, feature_cols, class_label
                )
                data.loc[class_indices] = imputed_class_data
        else:
            # Impute all data together
            data = self._impute_class(data, cols_to_impute, feature_cols, None)

        return data

    def _impute_class(
        self,
        data: pd.DataFrame,
        cols_to_impute: list,
        feature_cols: list,
        class_label: int | None,
    ) -> pd.DataFrame:
        if len(data) < self.n_neighbors:
            # Not enough samples - use global medians
            for col in cols_to_impute:
                zero_mask = data[col] == 0
                data.loc[zero_mask, col] = self.global_medians[col]
            return data

        # For each column that needs imputation
        for col in cols_to_impute:
            zero_indices = data[data[col] == 0].index
            if len(zero_indices) == 0:
                continue

            non_zero_indices = data[data[col] != 0].index
            if len(non_zero_indices) < self.n_neighbors:
                # Not enough non-zero samples - use global median
                data.loc[zero_indices, col] = self.global_medians[col]
                continue

            # Features for KNN: all except the current column being imputed
            knn_features = [c for c in feature_cols if c != col]

            # Prepare reference data (samples with non-zero values in target column)
            ref_data = data.loc[non_zero_indices, knn_features].copy()

            # Replace any zeros in reference features with median for KNN
            for feat in knn_features:
                if feat in cols_to_impute:
                    ref_data[feat] = ref_data[feat].replace(0, np.nan)
                    feat_median = ref_data[feat].dropna().median()
                    if pd.isna(feat_median):
                        feat_median = self.global_medians.get(feat, 0)
                    ref_data[feat] = ref_data[feat].fillna(feat_median)

            # Scale reference data
            scaler = self.scalers[class_label]

            # Only use the subset of features for this imputation
            ref_scaled = scaler.transform(data.loc[non_zero_indices, feature_cols])
            feat_indices = [feature_cols.index(f) for f in knn_features]
            ref_scaled = ref_scaled[:, feat_indices]

            # Fit KNN on reference data
            k = min(self.n_neighbors, len(ref_scaled))
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(ref_scaled)

            # For each sample with missing value
            for idx in zero_indices:
                query_data = data.loc[[idx], knn_features].copy()

                # Replace zeros in query features
                for feat in knn_features:
                    if feat in cols_to_impute and query_data.loc[idx, feat] == 0:
                        query_data.loc[idx, feat] = self.global_medians.get(feat, 0)

                # Scale query
                query_full = data.loc[[idx], feature_cols].copy()
                for feat in feature_cols:
                    if feat in cols_to_impute and query_full.loc[idx, feat] == 0:
                        query_full.loc[idx, feat] = self.global_medians.get(feat, 0)

                query_scaled = scaler.transform(query_full)
                query_scaled = query_scaled[:, feat_indices]

                # Find k nearest neighbors
                distances, indices = knn.kneighbors(query_scaled)
                neighbor_indices = non_zero_indices[indices[0]]

                # Get neighbor values for the target column
                neighbor_values = data.loc[neighbor_indices, col].values

                # Calculate imputed value
                if self.weights == 'distance':
                    # Weight by inverse distance (avoid division by zero)
                    weights = 1.0 / (distances[0] + 1e-10)
                    weights = weights / weights.sum()
                    imputed_value = np.average(neighbor_values, weights=weights)
                else:
                    # Uniform weights - just use median
                    imputed_value = np.median(neighbor_values)

                data.loc[idx, col] = imputed_value

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_stats(self) -> dict:
        if not self.fitted:
            raise ValueError("Imputer not fitted. Call fit() first.")

        return {
            "n_neighbors": self.n_neighbors,
            "class_aware": self.class_aware,
            "weights": self.weights,
            "random_state": self.random_state,
            "global_medians": {k: float(v) for k, v in self.global_medians.items()},
            "n_classes": len(self.scalers) if self.class_aware else 1,
        }


def impute_dataset_knn(
    input_path: Path,
    output_path: Path,
    n_neighbors: int = 5,
    random_state: int = 42,
    class_aware: bool = True,
) -> tuple[Path, KNNImputer]:
    df = pd.read_csv(input_path)

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        random_state=random_state,
        class_aware=class_aware,
        weights="distance",
    )
    imputed_df = imputer.fit_transform(df)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_df.to_csv(output_path, index=False)

    return output_path, imputer
