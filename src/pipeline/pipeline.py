
from pathlib import Path
import pandas as pd


class Pipeline:

    def __init__(self, train_path: Path, test_path: Path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None

    def load_data(self) -> "Pipeline":
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        return self

    def run_knn(self, n_neighbors: int = 5):
        raise NotImplementedError("KNN implementation coming soon")

    def run_kmeans(self, n_clusters: int = 2):
        raise NotImplementedError("K-means implementation coming soon")
