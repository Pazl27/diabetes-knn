
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .pipeline.splitter import split_dataset, get_split_stats
from .pipeline.predictor import run_prediction
from .algorithms.kmeans_imputer import impute_dataset, get_imputation_stats
import pandas as pd

console = Console()


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-s", "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
@click.option(
    "--no-impute",
    is_flag=True,
    default=False,
    help="Skip K-Means imputation (baseline mode)",
)
@click.version_option(version="0.1.0", prog_name="diabetes-pipeline")
def cli(input_file: Path, seed: int, no_impute: bool):
    """
    Diabetes Pipeline - KNN Classification with optional K-Means imputation.

    Takes a CSV file and runs the complete pipeline:
    1. Split into train/test sets (stratified)
    2. [Optional] Impute missing values using class-aware K-Means
    3. Train and evaluate KNN classifier with optimal hyperparameters

    \b
    Examples:
      # With K-Means imputation (recommended - best accuracy)
      python -m diabetes_pipeline.main data/diabetes.csv

      # Without imputation (baseline)
      python -m diabetes_pipeline.main data/diabetes.csv --no-impute
    """
    output = Path("data/processed")
    test_size = 0.2

    # Determine mode
    mode = "Baseline (No Imputation)" if no_impute else "K-Means + KNN (Recommended)"

    console.print(Panel.fit(
        f"[bold blue]Diabetes Pipeline v2.0[/bold blue]\n"
        f"Mode: [bold cyan]{mode}[/bold cyan]\n"
        f"Input: {input_file}\n"
        f"Output: {output}",
        title="Pipeline Configuration",
    ))

    try:
        # Step 1: Split dataset
        console.print("\n[bold]Step 1: Splitting dataset[/bold]")

        train_path, test_path = split_dataset(
            input_path=input_file,
            output_dir=output,
            test_size=test_size,
            target_column="Outcome",
            random_state=seed,
        )

        # Get and display split statistics
        stats = get_split_stats(train_path, test_path, "Outcome")

        table = Table(title="Split Statistics")
        table.add_column("Set", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Positive", justify="right", style="red")
        table.add_column("Negative", justify="right", style="green")
        table.add_column("Ratio", justify="right")

        for name, s in stats.items():
            table.add_row(
                name.capitalize(),
                str(s["total"]),
                str(s["positive"]),
                str(s["negative"]),
                f"{s['positive_ratio']:.2%}",
            )

        console.print(table)

        # Step 2: Impute missing values (optional)
        if no_impute:
            console.print("\n[bold yellow]Step 2: Skipping imputation (baseline mode)[/bold yellow]")
            console.print("ℹ️  Using raw data with zeros for missing values")
            train_imputed_path = train_path
            test_imputed_path = test_path
        else:
            console.print("\n[bold]Step 2: Imputing missing values[/bold]")
            console.print("Using class-aware K-Means clustering")

            # Impute train set
            train_imputed_path = output / "train_imputed.csv"
            original_train = pd.read_csv(train_path)

            _, imputer = impute_dataset(
                input_path=train_path,
                output_path=train_imputed_path,
                n_clusters=None,  # Auto-detect optimal k
                random_state=seed,
                class_aware=True,  # BEST STRATEGY: Class-aware clustering
            )

            # Get cluster stats
            cluster_stats = imputer.get_cluster_stats()

            # Display K-means info
            kmeans_table = Table(title="K-Means Clustering Info")
            kmeans_table.add_column("Parameter", style="cyan")
            kmeans_table.add_column("Value", justify="right")

            kmeans_table.add_row("Method", "Class-Aware K-Means")
            kmeans_table.add_row("Clusters", str(cluster_stats['n_clusters']))
            kmeans_table.add_row("Silhouette Score", f"{cluster_stats['silhouette']:.3f}")
            kmeans_table.add_row("Iterations", str(cluster_stats["n_iterations"]))
            kmeans_table.add_row("Inertia", f"{cluster_stats['inertia']:.2f}")
            kmeans_table.add_row("Random State", str(seed))

            console.print(kmeans_table)

            # Display detailed cluster breakdown per class
            if cluster_stats.get('n_clusters_per_class'):
                cluster_detail_table = Table(title="Clusters Per Class Detail")
                cluster_detail_table.add_column("Class", style="cyan")
                cluster_detail_table.add_column("Clusters", justify="right")
                cluster_detail_table.add_column("Total Samples", justify="right")
                cluster_detail_table.add_column("Avg per Cluster", justify="right")

                # Get original train data to count samples per class
                train_data = pd.read_csv(train_path)

                for class_key, n_clust in sorted(cluster_stats['n_clusters_per_class'].items()):
                    class_label = int(class_key.split('_')[1])
                    class_name = "Non-Diabetic" if class_label == 0 else "Diabetic    "

                    # Count total samples for this class
                    class_total = len(train_data[train_data['Outcome'] == class_label])
                    avg_per_cluster = class_total / n_clust if n_clust > 0 else 0

                    cluster_detail_table.add_row(
                        f"{class_name} ({class_label})",
                        str(n_clust),
                        str(class_total),
                        f"~{avg_per_cluster:.0f}"
                    )

                console.print(cluster_detail_table)

            # Display imputation statistics for train
            imputed_train = pd.read_csv(train_imputed_path)
            imp_stats = get_imputation_stats(original_train, imputed_train)

            imp_table = Table(title="Imputation Statistics (Train)")
            imp_table.add_column("Column", style="cyan")
            imp_table.add_column("Zeros Replaced", justify="right")
            imp_table.add_column("Orig Median", justify="right")
            imp_table.add_column("New Median", justify="right")

            for col, s in imp_stats.items():
                imp_table.add_row(
                    col,
                    str(s["zeros_replaced"]),
                    f"{s['original_median']:.1f}",
                    f"{s['imputed_median']:.1f}",
                )

            console.print(imp_table)

            # Impute test set
            console.print("Applying imputation to test set...")
            test_imputed_path = output / "test_imputed.csv"
            impute_dataset(
                input_path=test_path,
                output_path=test_imputed_path,
                n_clusters=None,
                random_state=seed,
                class_aware=True,
            )

        # Step 3: KNN Classification
        console.print("\n[bold]Step 3: KNN Classification[/bold]")
        console.print("Using optimized hyperparameters:")
        console.print("  • 5-fold stratified cross-validation")
        console.print("  • Distance-weighted predictions")
        console.print("  • F1 score optimization")
        console.print("  • Auto k-selection (odd values 3-29)")

        predictions_path = output / "predictions.csv"
        classifier, results = run_prediction(
            train_path=train_imputed_path,
            test_path=test_imputed_path,
            output_path=predictions_path,
            random_state=seed,
        )

        # Display KNN info
        knn_stats = classifier.get_stats()
        knn_table = Table(title="KNN Classifier Info")
        knn_table.add_column("Parameter", style="cyan")
        knn_table.add_column("Value", justify="right")

        k_label = f"{knn_stats['n_neighbors']} (auto-selected)" if knn_stats["auto_k"] else str(knn_stats['n_neighbors'])
        knn_table.add_row("Neighbors (k)", k_label)
        if knn_stats["auto_k"] and knn_stats.get("best_k_score"):
            knn_table.add_row("CV F1 Score", f"{knn_stats['best_k_score']:.3f}")
        knn_table.add_row("Distance Metric", knn_stats.get("metric", "euclidean"))
        knn_table.add_row("Total Features", str(len(knn_stats["feature_columns"])))
        knn_table.add_row("Weight Method", "distance")

        console.print(knn_table)

        # Display feature importances if available
        if knn_stats.get("feature_importances"):
            feat_table = Table(title="Top Feature Importances (Mutual Information)")
            feat_table.add_column("Feature", style="cyan")
            feat_table.add_column("Score", justify="right")

            sorted_feats = sorted(
                knn_stats["feature_importances"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5

            for feat, score in sorted_feats:
                feat_table.add_row(feat, f"{score:.4f}")

            console.print(feat_table)

        # Display evaluation results
        eval_table = Table(title="Evaluation Results (Test Set)")
        eval_table.add_column("Metric", style="cyan")
        eval_table.add_column("Value", justify="right")

        eval_table.add_row("Accuracy", f"[bold]{results['accuracy']:.2%}[/bold]")
        eval_table.add_row("Precision", f"{results['precision']:.2%}")
        eval_table.add_row("Recall", f"{results['recall']:.2%}")
        eval_table.add_row("F1 Score", f"{results['f1']:.2%}")

        console.print(eval_table)

        # Display confusion matrix
        cm = results["confusion_matrix"]
        cm_table = Table(title="Confusion Matrix")
        cm_table.add_column("", style="bold")
        cm_table.add_column("Pred: No", justify="right")
        cm_table.add_column("Pred: Yes", justify="right")

        cm_table.add_row("Actual: No", str(cm["tn"]), str(cm["fp"]))
        cm_table.add_row("Actual: Yes", str(cm["fn"]), str(cm["tp"]))

        console.print(cm_table)

        # Summary
        console.print("\n" + "="*60)
        console.print("[bold green]Pipeline completed successfully![/bold green]")
        console.print("="*60)

        console.print(f"\nOutput files:")
        console.print(f"  • Train split: {train_path}")
        console.print(f"  • Test split: {test_path}")
        if not no_impute:
            console.print(f"  • Train imputed: {train_imputed_path}")
            console.print(f"  • Test imputed: {test_imputed_path}")
        console.print(f"  • Predictions: {predictions_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
