import os
import time
import csv
import statistics
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import torch
import yaml
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from pathlib import Path

def train_sign_language_yolo():
    """Train YOLOv8n across augmented and non-augmented paradigms."""
    
    print("Starting YOLO Sign Language Detection Training...")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dataset configuration
    dataset_path = "./yolo_dataset4/dataset.yaml"
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset configuration not found: {dataset_path}")
    
    # Load dataset info
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"Dataset: {dataset_config['nc']} classes")
    print(f"Classes: {dataset_config['names'][:10]}..." if len(dataset_config['names']) > 10 else f"Classes: {dataset_config['names']}")
    
    # Initialize YOLO model
    print("\nInitializing YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # nano version for speed
    
    # Training parameters
    seeds = [42, 17, 119]
    batch_sizes = [8, 16, 32]
    paradigms = [
        ('non_augmented', False, 'sign_lang_detection_non_augmented'),
    ]
    metric_names = ['map50', 'map50_95', 'precision', 'recall', 'f1']

    for paradigm, use_augmentation, project in paradigms:
        run_results = []
        print(f"\n===== {paradigm.upper()} YOLOV8N EXPERIMENTS =====")

        for seed in seeds:
            for batch_size in batch_sizes:
                run_name = f'yolov8n_seed_{seed}_batch_{batch_size}'
                print(f"\nStarting {paradigm} run: seed={seed}, batch={batch_size}")
                model = YOLO('yolov8n.pt')
                training_args = {
                    'data': dataset_path,
                    'epochs': 10,
                    'imgsz': 640,
                    'batch': batch_size,
                    'device': device,
                    'project': project,
                    'name': run_name,
                    'exist_ok': False,
                    'plots': True,
                    'val': True,
                    'seed': seed,
                }
                if not use_augmentation:
                    training_args.update({
                        'hsv_h': 0.0,
                        'hsv_s': 0.0,
                        'hsv_v': 0.0,
                        'degrees': 0.0,
                        'translate': 0.0,
                        'scale': 0.0,
                        'shear': 0.0,
                        'perspective': 0.0,
                        'flipud': 0.0,
                        'fliplr': 0.0,
                        'mosaic': 0.0,
                        'mixup': 0.0,
                        'cutmix': 0.0,
                        'copy_paste': 0.0,
                    })

                try:
                    results = model.train(**training_args)
                    best_model_path = results.save_dir / 'weights' / 'best.pt'
                    best_model = YOLO(str(best_model_path))
                    val_results = best_model.val(data=dataset_path, imgsz=640, split='val')
                    precision = float(val_results.box.mp)
                    recall = float(val_results.box.mr)
                    metrics = {
                        'paradigm': paradigm,
                        'seed': seed,
                        'batch_size': batch_size,
                        'map50': float(val_results.box.map50),
                        'map50_95': float(val_results.box.map),
                        'precision': precision,
                        'recall': recall,
                        'f1': 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
                    }
                    run_results.append(metrics)
                    print(
                        f"Run seed={seed}, batch={batch_size}: "
                        f"mAP50={metrics['map50']:.4f}, "
                        f"mAP50-95={metrics['map50_95']:.4f}, "
                        f"Precision={metrics['precision']:.4f}, "
                        f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
                    )
                except Exception as error:
                    print(f"Run failed for seed={seed}, batch={batch_size}: {error}")

        if len(run_results) != len(seeds) * len(batch_sizes):
            raise RuntimeError(
                f"Expected {len(seeds) * len(batch_sizes)} completed runs for {paradigm}, "
                f"but got {len(run_results)}."
            )

        output_dir = project
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'yolov8_results.csv'), 'w', newline='') as results_file:
            writer = csv.DictWriter(results_file, fieldnames=['paradigm', 'seed', 'batch_size', *metric_names])
            writer.writeheader()
            writer.writerows(run_results)

        with open(os.path.join(output_dir, 'yolov8_summary.csv'), 'w', newline='') as summary_file:
            writer = csv.DictWriter(
                summary_file,
                fieldnames=['group', 'metric', 'mean', 'sample_standard_deviation', 'runs'],
            )
            writer.writeheader()
            groups = [('All runs', run_results)]
            groups.extend(
                (
                    f'Batch size {batch_size}',
                    [run for run in run_results if run['batch_size'] == batch_size],
                )
                for batch_size in batch_sizes
            )
            for group, group_runs in groups:
                print(f"\n{paradigm} {group} summary (mean +/- sample standard deviation):")
                for metric in metric_names:
                    values = [run[metric] for run in group_runs]
                    mean = statistics.mean(values)
                    standard_deviation = statistics.stdev(values)
                    writer.writerow({
                        'group': group,
                        'metric': metric,
                        'mean': mean,
                        'sample_standard_deviation': standard_deviation,
                        'runs': len(values),
                    })
                    print(f"{metric}: {mean:.4f} +/- {standard_deviation:.4f}")

    print("\nAll YOLOv8n experiments completed successfully!")
    return None, None

def visualize_training_metrics(save_dir):
    """Visualize training & validation metrics using results.csv"""
    
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        print(f"No results.csv found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Training loss curves
    plt.figure(figsize=(12,6))
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid()
    plt.savefig(Path(save_dir) / "train_losses.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Validation metrics
    plt.figure(figsize=(12,6))
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    plt.grid()
    plt.savefig(Path(save_dir) / "val_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Custom training visualizations saved in {save_dir}")

def validate_model(model_path, dataset_path):
    """Validate the trained model"""
    
    print("\nValidating trained model...")
    print("-" * 40)
    
    try:
        model = YOLO(model_path)
        val_results = model.val(data=dataset_path, imgsz=640)
        
        print("\nValidation Results:")
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"mAP50-95: {val_results.box.map:.4f}")
        
        return val_results
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return None

def create_inference_script():
    """Create an inference script for real-time detection"""
    
    pass

def main():
    print("YOLO Sign Language Detection Training Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    train_sign_language_yolo()
    end_time = time.time()
    elapsed_time = end_time - start_time
    tot_hours = elapsed_time / 3600
    print(f"\nTotal training time: ({tot_hours:.2f} hours)")
    create_inference_script()
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
