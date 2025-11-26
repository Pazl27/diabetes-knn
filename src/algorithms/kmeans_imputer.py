import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path


class KMeansImputer:
    # Columns where 0 is not a valid value and should be imputed
    COLUMNS_TO_IMPUTE = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    def __init__(self, n_clusters: int | None = None, random_state: int = 42, class_aware: bool = True):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.class_aware = class_aware
        self.kmeans_models = {}  # One per class if class_aware
        self.scalers = {}  # One per class if class_aware
        self.cluster_medians = {}
        self.cluster_counts = {}
        self.cluster_features = []
        self.inertia = None
        self.silhouette = None
        self.auto_k = n_clusters is None
        self.global_medians = {}  # Fallback for test data

    def _find_optimal_k(self, scaled_features: np.ndarray, k_range: range = range(2, 9)) -> int:
        if len(scaled_features) < 2:
            return 2

        best_k = 2
        best_score = -1

        max_k = min(max(k_range), len(scaled_features) - 1)
        actual_k_range = range(max(2, min(k_range)), max_k + 1)

        for k in actual_k_range:
            if k >= len(scaled_features):
                break
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300,
                )
                labels = kmeans.fit_predict(scaled_features)

                # Only calculate silhouette if we have multiple unique labels
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(scaled_features, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception:
                continue

        return best_k

    def fit(self, df: pd.DataFrame) -> "KMeansImputer":
        # Work with a copy
        data = df.copy()

        # Get columns that exist in the data
        cols_to_impute = [c for c in self.COLUMNS_TO_IMPUTE if c in data.columns]

        # Calculate global medians as fallback
        for col in cols_to_impute:
            non_zero_values = data[data[col] != 0][col]
            self.global_medians[col] = non_zero_values.median() if len(non_zero_values) > 0 else data[col].median()

        # Replace 0s with NaN for imputation columns
        for col in cols_to_impute:
            data[col] = data[col].replace(0, np.nan)

        # Use features for clustering (exclude target)
        cluster_features = [c for c in data.columns if c != "Outcome"]
        self.cluster_features = cluster_features

        if self.class_aware and "Outcome" in data.columns:
            # Cluster each class separately
            classes = data["Outcome"].unique()
            total_inertia = 0
            total_samples = 0
            all_silhouettes = []

            for class_label in sorted(classes):
                class_data = data[data["Outcome"] == class_label].copy()

                # For clustering: replace NaN with column median
                cluster_data = class_data[cluster_features].copy()
                for col in cluster_features:
                    col_median = cluster_data[col].dropna().median()
                    if pd.isna(col_median):
                        col_median = self.global_medians.get(col, 0)
                    cluster_data[col] = cluster_data[col].fillna(col_median)

                # Scale features for this class
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(cluster_data)
                self.scalers[class_label] = scaler

                # Find optimal k for this class if not specified
                n_clusters_for_class = self.n_clusters
                if n_clusters_for_class is None:
                    n_clusters_for_class = self._find_optimal_k(scaled_features)

                # Fit K-means for this class
                kmeans = KMeans(
                    n_clusters=n_clusters_for_class,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300,
                )
                clusters = kmeans.fit_predict(scaled_features)
                self.kmeans_models[class_label] = kmeans

                # Track metrics
                total_inertia += kmeans.inertia_
                total_samples += len(scaled_features)

                if len(np.unique(clusters)) > 1:
                    class_silhouette = silhouette_score(scaled_features, clusters)
                    all_silhouettes.append(class_silhouette)

                class_data["_cluster"] = clusters

                # Count samples per cluster
                for cluster_id in range(n_clusters_for_class):
                    cluster_key = (class_label, cluster_id)
                    self.cluster_counts[cluster_key] = int((clusters == cluster_id).sum())

                # Calculate median for each column in each cluster
                for col in cols_to_impute:
                    if col not in self.cluster_medians:
                        self.cluster_medians[col] = {}

                    for cluster_id in range(n_clusters_for_class):
                        cluster_mask = class_data["_cluster"] == cluster_id
                        cluster_values = class_data[cluster_mask][col].dropna()

                        # Get median of non-NaN values in this cluster
                        if len(cluster_values) > 0:
                            median_val = cluster_values.median()
                        else:
                            # Fallback to class median
                            class_values = class_data[col].dropna()
                            median_val = class_values.median() if len(class_values) > 0 else self.global_medians[col]

                        cluster_key = (class_label, cluster_id)
                        self.cluster_medians[col][cluster_key] = median_val

            self.inertia = total_inertia / total_samples if total_samples > 0 else 0
            self.silhouette = np.mean(all_silhouettes) if all_silhouettes else 0

        else:
            # Original non-class-aware clustering
            cluster_data = data[cluster_features].copy()
            for col in cluster_features:
                col_median = cluster_data[col].dropna().median()
                if pd.isna(col_median):
                    col_median = self.global_medians.get(col, 0)
                cluster_data[col] = cluster_data[col].fillna(col_median)

            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_data)
            self.scalers[None] = scaler

            # Find optimal k if not specified
            if self.n_clusters is None:
                self.n_clusters = self._find_optimal_k(scaled_features)

            # Fit K-means
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )
            clusters = kmeans.fit_predict(scaled_features)
            self.kmeans_models[None] = kmeans
            data["_cluster"] = clusters

            self.inertia = kmeans.inertia_
            self.silhouette = silhouette_score(scaled_features, clusters)

            # Count samples per cluster
            for cluster_id in range(self.n_clusters):
                self.cluster_counts[cluster_id] = int((clusters == cluster_id).sum())

            # Calculate median for each column in each cluster
            for col in cols_to_impute:
                self.cluster_medians[col] = {}
                for cluster_id in range(self.n_clusters):
                    cluster_data = data[data["_cluster"] == cluster_id][col]
                    median_val = cluster_data.dropna().median()

                    # Fallback to global median if cluster has no valid values
                    if pd.isna(median_val):
                        median_val = self.global_medians[col]

                    self.cluster_medians[col][cluster_id] = median_val

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.kmeans_models:
            raise ValueError("Imputer not fitted. Call fit() first.")

        data = df.copy()
        cols_to_impute = [c for c in self.COLUMNS_TO_IMPUTE if c in data.columns]

        # Convert columns to float to avoid dtype warnings
        for col in cols_to_impute:
            data[col] = data[col].astype(float)

        if self.class_aware and "Outcome" in data.columns:
            # Transform each class separately
            for class_label in data["Outcome"].unique():
                if class_label not in self.kmeans_models:
                    # Unknown class in test data - use global medians
                    class_mask = data["Outcome"] == class_label
                    for col in cols_to_impute:
                        zero_mask = (data[col] == 0) & class_mask
                        data.loc[zero_mask, col] = self.global_medians[col]
                    continue

                class_mask = data["Outcome"] == class_label
                class_data = data[class_mask].copy()

                # Prepare cluster data (same as fit)
                cluster_data = class_data[self.cluster_features].copy()
                for col in cols_to_impute:
                    cluster_data[col] = cluster_data[col].replace(0, np.nan)
                    col_median = cluster_data[col].dropna().median()
                    if pd.isna(col_median):
                        col_median = self.global_medians.get(col, 0)
                    cluster_data[col] = cluster_data[col].fillna(col_median)

                # Predict clusters
                scaler = self.scalers[class_label]
                scaled_features = scaler.transform(cluster_data)
                kmeans = self.kmeans_models[class_label]
                clusters = kmeans.predict(scaled_features)

                # Impute values
                for idx, cluster_id in enumerate(clusters):
                    original_idx = class_data.index[idx]
                    for col in cols_to_impute:
                        if data.loc[original_idx, col] == 0:
                            cluster_key = (class_label, cluster_id)
                            if cluster_key in self.cluster_medians[col]:
                                data.loc[original_idx, col] = self.cluster_medians[col][cluster_key]
                            else:
                                data.loc[original_idx, col] = self.global_medians[col]

        else:
            # Non-class-aware transform
            cluster_data = data.copy()
            for col in cols_to_impute:
                cluster_data[col] = cluster_data[col].replace(0, np.nan)
                col_median = cluster_data[col].dropna().median()
                if pd.isna(col_median):
                    col_median = self.global_medians.get(col, 0)
                cluster_data[col] = cluster_data[col].fillna(col_median)

            # Predict clusters
            scaler = self.scalers[None]
            scaled_features = scaler.transform(cluster_data[self.cluster_features])
            kmeans = self.kmeans_models[None]
            clusters = kmeans.predict(scaled_features)

            # Impute values
            for idx, cluster_id in enumerate(clusters):
                for col in cols_to_impute:
                    if data.iloc[idx][col] == 0:
                        data.iloc[idx, data.columns.get_loc(col)] = self.cluster_medians[col][cluster_id]

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_cluster_stats(self) -> dict:
        """Get statistics about the K-means clustering."""
        if not self.kmeans_models:
            raise ValueError("Imputer not fitted. Call fit() first.")

        if self.class_aware:
            n_clusters_info = {}
            for class_label, kmeans in self.kmeans_models.items():
                n_clusters_info[f"class_{class_label}"] = kmeans.n_clusters
            n_clusters_display = f"{len(self.kmeans_models)} classes"
        else:
            n_clusters_display = str(self.n_clusters)

        return {
            "n_clusters": n_clusters_display,
            "n_clusters_per_class": n_clusters_info if self.class_aware else {},
            "class_aware": self.class_aware,
            "cluster_features": self.cluster_features,
            "cluster_counts": {str(k): v for k, v in self.cluster_counts.items()},
            "inertia": float(self.inertia),
            "silhouette": float(self.silhouette),
            "n_iterations": max(km.n_iter_ for km in self.kmeans_models.values()),
            "random_state": self.random_state,
            "auto_k": self.auto_k,
        }


def impute_dataset(
    input_path: Path,
    output_path: Path,
    n_clusters: int | None = None,
    random_state: int = 42,
    class_aware: bool = True,
) -> tuple[Path, KMeansImputer]:
    df = pd.read_csv(input_path)

    imputer = KMeansImputer(n_clusters=n_clusters, random_state=random_state, class_aware=class_aware)
    imputed_df = imputer.fit_transform(df)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_df.to_csv(output_path, index=False)

    return output_path, imputer


def get_imputation_stats(original_df: pd.DataFrame, imputed_df: pd.DataFrame) -> dict:
    stats = {}

    for col in KMeansImputer.COLUMNS_TO_IMPUTE:
        if col in original_df.columns:
            zeros_count = (original_df[col] == 0).sum()
            stats[col] = {
                "zeros_replaced": int(zeros_count),
                "original_mean": float(original_df[col].replace(0, np.nan).mean()),
                "imputed_mean": float(imputed_df[col].mean()),
                "original_median": float(original_df[col].replace(0, np.nan).median()),
                "imputed_median": float(imputed_df[col].median()),
            }

    return stats
