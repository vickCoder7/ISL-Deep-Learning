"""Run VGG16 classification on the same variable-box image split as YOLOv8n."""

import argparse
import csv
import os
import random
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "yolo_models" / "yolo_variable_boxes"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Variable-box dataset directory containing annotations.csv and train/val.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 119])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--project", type=Path, default=Path("./variable_vgg16_results"))
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_generators(data_dir, batch_size, seed):
    annotations = pd.read_csv(data_dir / "annotations.csv")
    annotations["image_path"] = annotations.apply(
        lambda row: str(data_dir / row["split"] / "images" / row["image"]), axis=1
    )
    annotations["class_name"] = annotations["class_name"].astype(str)

    train_frame = annotations[annotations["split"] == "train"].copy()
    validation_frame = annotations[annotations["split"] == "val"].copy()

    # Match the no-augmentation comparison condition: only normalize pixels.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        train_frame,
        x_col="image_path",
        y_col="class_name",
        classes=CLASSES,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
    )
    validation_generator = validation_datagen.flow_from_dataframe(
        validation_frame,
        x_col="image_path",
        y_col="class_name",
        classes=CLASSES,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_generator, validation_generator


def build_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate(model, validation_generator):
    validation_generator.reset()
    probabilities = model.predict(validation_generator, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = validation_generator.classes
    return {
        "accuracy": float(np.mean(predicted == actual)),
        "precision": float(precision_score(actual, predicted, average="macro", zero_division=0)),
        "recall": float(recall_score(actual, predicted, average="macro", zero_division=0)),
        "f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def main():
    args = parse_args()
    annotations_path = args.data / "annotations.csv"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation metadata not found: {annotations_path}")

    args.project.mkdir(parents=True, exist_ok=True)
    run_metrics = []

    for seed in args.seeds:
        for batch_size in args.batch_sizes:
            set_seed(seed)
            run_name = f"vgg16_seed_{seed}_batch_{batch_size}"
            run_dir = args.project / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nStarting run with seed {seed}, batch size {batch_size}...")

            train_generator, validation_generator = create_generators(args.data, batch_size, seed)
            model = build_model()
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7, verbose=1),
                ModelCheckpoint(
                    run_dir / "best_vgg16.keras",
                    monitor="val_accuracy",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                ),
            ]
            model.fit(
                train_generator,
                epochs=args.epochs,
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1,
            )

            metrics = {"seed": seed, "batch_size": batch_size, **evaluate(model, validation_generator)}
            run_metrics.append(metrics)
            print(
                f"Run seed={seed}, batch={batch_size}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
            )
            tf.keras.backend.clear_session()

    metric_names = ["accuracy", "precision", "recall", "f1"]
    summary_rows = []

    def summarize(label, runs):
        print(f"\n{label} summary (mean +/- sample standard deviation):")
        for metric_name in metric_names:
            values = [run[metric_name] for run in runs]
            mean = statistics.mean(values)
            standard_deviation = statistics.stdev(values)
            summary_rows.append({
                "group": label,
                "metric": metric_name,
                "mean": mean,
                "sample_standard_deviation": standard_deviation,
                "runs": len(values),
            })
            print(f"{metric_name}: {mean:.4f} +/- {standard_deviation:.4f}")

    summarize("All runs", run_metrics)
    for batch_size in args.batch_sizes:
        summarize(
            f"Batch size {batch_size}",
            [run for run in run_metrics if run["batch_size"] == batch_size],
        )

    with (args.project / "vgg16_variable_boxes_results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["seed", "batch_size", *metric_names])
        writer.writeheader()
        writer.writerows(run_metrics)

    with (args.project / "vgg16_variable_boxes_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["group", "metric", "mean", "sample_standard_deviation", "runs"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
