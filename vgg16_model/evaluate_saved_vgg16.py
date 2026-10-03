"""Evaluate saved VGG16 runs and add classification metrics to their CSV files."""

import csv
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "datasets" / "final" / "train"
SEEDS = [42, 17, 119]
BATCH_SIZES = [8, 16, 32]
PARADIGMS = {
    "non_augmented": ROOT / "models_non_augm_data",
    "augmented": ROOT / "models_augm_data",
}


def validation_generator(batch_size, seed):
    generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )
    return generator.flow_from_directory(
        DATA_DIR,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=seed,
    )


def evaluate_run(model_path, batch_size, seed):
    generator = validation_generator(batch_size, seed)
    model = tf.keras.models.load_model(model_path)
    generator.reset()
    probabilities = model.predict(generator, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = generator.classes
    return {
        "precision": float(precision_score(actual, predicted, average="macro", zero_division=0)),
        "recall": float(recall_score(actual, predicted, average="macro", zero_division=0)),
        "f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def write_summary(result_directory, rows):
    metric_names = [
        "train_accuracy",
        "validation_accuracy",
        "train_loss",
        "validation_loss",
        "training_time_seconds",
        "precision",
        "recall",
        "f1",
    ]
    summary_rows = []
    groups = [("All runs", rows)]
    groups.extend(
        (
            f"Batch size {batch_size}",
            [row for row in rows if int(row["batch_size"]) == batch_size],
        )
        for batch_size in BATCH_SIZES
    )
    for group, group_rows in groups:
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows]
            summary_rows.append({
                "group": group,
                "metric": metric,
                "mean": statistics.mean(values),
                "sample_standard_deviation": statistics.stdev(values),
                "runs": len(values),
            })

    with (result_directory / "vgg16_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["group", "metric", "mean", "sample_standard_deviation", "runs"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    for paradigm, result_directory in PARADIGMS.items():
        results_path = result_directory / "vgg16_results.csv"
        rows = pd.read_csv(results_path).to_dict("records")
        print(f"\nEvaluating {paradigm} saved models...")
        for row in rows:
            seed = int(row["seed"])
            batch_size = int(row["batch_size"])
            model_path = result_directory / f"batch_{batch_size}" / f"seed_{seed}" / "vgg16_model.keras"
            metrics = evaluate_run(model_path, batch_size, seed)
            row.update(metrics)
            print(
                f"{paradigm}, seed={seed}, batch={batch_size}: "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
            )

        fieldnames = list(rows[0].keys())
        with results_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        write_summary(result_directory, rows)
        print(f"Updated {results_path} and {result_directory / 'vgg16_summary.csv'}")


if __name__ == "__main__":
    main()
