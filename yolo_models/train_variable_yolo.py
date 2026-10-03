"""Train and validate YOLOv8n on the manually annotated variable-box set."""

import argparse
import csv
import statistics
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("./yolo_variable_boxes/dataset.yaml"))
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 17, 119])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="sign_lang_detection_variable_boxes")
    parser.add_argument("--name", default="yolov8n_manual_variable_boxes")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}. "
            "Run annotate_variable_boxes.py first."
        )

    run_metrics = []

    for seed in args.seeds:
        for batch_size in args.batch_sizes:
            run_name = f"{args.name}_seed_{seed}_batch_{batch_size}"
            print(f"\nStarting run with seed {seed}, batch size {batch_size}...")
            model = YOLO(args.weights)
            model.train(
                data=str(args.data),
                epochs=args.epochs,
                batch=batch_size,
                imgsz=args.imgsz,
                project=args.project,
                name=run_name,
                exist_ok=False,
                plots=True,
                val=True,
                seed=seed,
            )

            best_weights = Path(args.project) / run_name / "weights" / "best.pt"
            if not best_weights.exists():
                raise FileNotFoundError(f"Best checkpoint not found: {best_weights}")

            best_model = YOLO(str(best_weights))
            results = best_model.val(data=str(args.data), imgsz=args.imgsz, split="val")
            precision = float(results.box.mp)
            recall = float(results.box.mr)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            metrics = {
                "seed": seed,
                "batch_size": batch_size,
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            run_metrics.append(metrics)
            print(
                f"Run seed={seed}, batch={batch_size}: "
                f"mAP50={metrics['map50']:.4f}, "
                f"mAP50-95={metrics['map50_95']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}"
            )

    metric_names = ["map50", "map50_95", "precision", "recall", "f1"]
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
        batch_runs = [run for run in run_metrics if run["batch_size"] == batch_size]
        summarize(f"Batch size {batch_size}", batch_runs)

    output_dir = Path(args.project)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.name}_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["seed", "batch_size", *metric_names])
        writer.writeheader()
        writer.writerows(run_metrics)

    with (output_dir / f"{args.name}_summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["group", "metric", "mean", "sample_standard_deviation", "runs"])
        for row in summary_rows:
            writer.writerow([
                row["group"],
                row["metric"],
                row["mean"],
                row["sample_standard_deviation"],
                row["runs"],
            ])


if __name__ == "__main__":
    main()