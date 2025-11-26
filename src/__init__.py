
from .pipeline.splitter import split_dataset
from .pipeline.pipeline import Pipeline
from .algorithms.kmeans_imputer import KMeansImputer

__version__ = "0.1.0"
__all__ = ["split_dataset", "Pipeline", "KMeansImputer"]

def main():
    from .cli import cli
    cli()
