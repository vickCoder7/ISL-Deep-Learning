import os
import time
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
    """Train YOLOv8 model for sign language detection"""
    
    print("Starting YOLO Sign Language Detection Training...")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dataset configuration
    dataset_path = "./yolo_dataset/dataset.yaml"
    
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
    training_args = {
        'data': dataset_path,
        'epochs': 10,
        'imgsz': 640,
        'batch': 32,
        'device': device,
        'project': 'sign_lang_detection_diff_params',
        'name': 'yolov8_sign_detection_rs42_10-32', # epochs-10-batch-32
        'exist_ok': True,
        'plots': True,
        'val': True,
        'seed': 42
    }
    
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        results = model.train(**training_args)
        
        print("\nTraining completed successfully!")
        print("=" * 60)
        print(f"Training results saved in: sign_lang_detection_diff_params/yolov8_sign_detection_rs42_10-32")
        
        # Get the best model path
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        print(f"Best model saved at: {best_model_path}")
        
        # Create visualizations
        visualize_training_metrics(results.save_dir)
        
        return best_model_path, results
        
    except Exception as e:
        print(f"Error during training: {e}")
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
    
    # Train the model
    start_time = time.time()

    best_model_path, results = train_sign_language_yolo()

    end_time = time.time()
    elapsed_time = end_time - start_time
    tot_hours = elapsed_time / 3600
    print(f"\nTotal training time: ({tot_hours:.2f} hours)")
    
    if best_model_path:
        validate_model(best_model_path, "./yolo_dataset/dataset.yaml")
        create_inference_script()
        
        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print("Training failed.")

if __name__ == "__main__":
    main()
