import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class KNNClassifier:

    def __init__(
        self,
        n_neighbors: int | None = None,
        random_state: int = 42,
        metric: str = "euclidean",
        n_features: int | None = None,
        cv_folds: int = 5,
    ):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.metric = metric
        self.n_features = n_features
        self.cv_folds = cv_folds
        self.knn = None
        self.scaler = None
        self.feature_selector = None
        self.feature_columns = []
        self.selected_features = []
        self.target_column = "Outcome"
        self.auto_k = n_neighbors is None
        self.k_scores = {}
        self.feature_importances = {}

    def _find_optimal_k(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_range: range = range(3, 31, 2),
    ) -> int:
        best_k = 5
        best_score = 0

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, metric=self.metric)
            # Use cross-validation on training set only
            scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='f1')
            mean_score = scores.mean()
            self.k_scores[k] = float(mean_score)

            if mean_score > best_score:
                best_score = mean_score
                best_k = k

        return best_k

    def _select_features(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.n_features is None or self.n_features >= X_train.shape[1]:
            self.selected_features = self.feature_columns
            return

        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X_train, y_train, random_state=self.random_state)

        # Store feature importances
        for i, col in enumerate(self.feature_columns):
            self.feature_importances[col] = float(mi_scores[i])

        # Select top k features
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif,
            k=self.n_features
        )
        self.feature_selector.fit(X_train, y_train)

        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [self.feature_columns[i] for i in selected_indices]

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "KNNClassifier":
        # Identify feature columns (all except target)
        self.feature_columns = [c for c in train_df.columns if c != self.target_column]

        X_train = train_df[self.feature_columns].values
        y_train = train_df[self.target_column].values

        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Feature selection
        self._select_features(X_train_scaled, y_train)

        if self.feature_selector is not None:
            X_train_selected = self.feature_selector.transform(X_train_scaled)
        else:
            X_train_selected = X_train_scaled

        # Find optimal k if not specified
        if self.n_neighbors is None:
            self.n_neighbors = self._find_optimal_k(X_train_selected, y_train)

        # Fit final KNN model
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            weights='distance',  # Weight by inverse distance for better predictions
        )
        self.knn.fit(X_train_selected, y_train)

        return self

    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        return X_selected

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.knn is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        X = df[self.feature_columns].values
        X_processed = self._preprocess_features(X)
        return self.knn.predict(X_processed)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.knn is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        X = df[self.feature_columns].values
        X_processed = self._preprocess_features(X)
        return self.knn.predict_proba(X_processed)

    def evaluate(self, df: pd.DataFrame) -> dict:
        y_true = df[self.target_column].values
        y_pred = self.predict(df)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "n_neighbors": self.n_neighbors,
            "auto_k": self.auto_k,
        }

    def get_stats(self) -> dict:
        if self.knn is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        return {
            "n_neighbors": self.n_neighbors,
            "auto_k": self.auto_k,
            "metric": self.metric,
            "feature_columns": self.feature_columns,
            "selected_features": self.selected_features,
            "n_features_selected": len(self.selected_features),
            "feature_importances": self.feature_importances,
            "k_scores": self.k_scores,
            "best_k_score": self.k_scores.get(self.n_neighbors, 0.0) if self.k_scores else 0.0,
        }
