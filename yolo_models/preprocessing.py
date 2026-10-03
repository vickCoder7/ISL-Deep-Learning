"""
    ./yolo_dataset; ./yolo_dataset4: random_state=42
    ./yolo_dataset2: random_state=17
    ./yolo_dataset3: random_state=119
"""
import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import yaml
import random
from pathlib import Path
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

class SignLanguageYOLOPreprocessor:
    def __init__(self, source_dir="../datasets/final/train", output_dir="./yolo_dataset4", img_size=640):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.img_size = img_size
        
        # Create class mapping for YOLO
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                       'U', 'V', 'W', 'X', 'Y', 'Z']
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"\nInitialized preprocessor with {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def create_yolo_structure(self):
        """Create YOLO dataset directory structure"""
        
        # Create main directories
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                Path(f"{self.output_dir}/{split}/{subdir}").mkdir(parents=True, exist_ok=True)
        
        print(f"Created YOLO directory structure in {self.output_dir}")
    
    def preprocess_image(self, image_path, target_size=None):
        """
        Preprocess image for YOLO format WITHOUT augmentation
        - Resize to target size while maintaining aspect ratio
        - Pad with gray pixels if needed
        """
        if target_size is None:
            target_size = self.img_size
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        original_height, original_width = img.shape[:2]
        
        # Calculate scaling factor to fit within target size
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # Gray padding
        
        # Calculate padding offsets to center the image
        y_offset = (target_size - new_height) // 2
        x_offset = (target_size - new_width) // 2
        
        # Place resized image in center
        img_padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized
        return img_padded, scale, x_offset, y_offset

    def create_yolo_annotation(self, class_name, img_width, img_height):
        """
        Create YOLO format annotation for the entire image
        Since these are classification images, we'll create a bounding box for the whole image
        """
        class_idx = self.class_to_idx[class_name]
        
        # For classification images, create a bounding box covering most of the image
        # YOLO format: class_id center_x center_y width height (all normalized to 0-1)
        center_x = 0.5
        center_y = 0.5
        bbox_width = 0.9   # Cover 90% of image width
        bbox_height = 0.9  # Cover 90% of image height
        
        return f"{class_idx} {center_x} {center_y} {bbox_width} {bbox_height}"

    
    def process_dataset(self, train_ratio=0.8, val_ratio=0.2):
        """
        Process the entire dataset and split into train/validation sets
        PRESERVES ORIGINAL DATA - augmentation will be applied during training
        """
        self.create_yolo_structure()
        
        all_files = []
        
        # Collect all image files with their class labels
        for class_name in self.classes:
            class_path = os.path.normpath(os.path.join(self.source_dir, class_name))
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        all_files.append((os.path.normpath(os.path.join(class_path, img_file)), class_name))
        
        print(f"Found {len(all_files)} images total")
        
        # Split into train and validation sets
        train_files, val_files = train_test_split(
            all_files, train_size=train_ratio, random_state=42, stratify=[f[1] for f in all_files]
        )
        
        print(f"Train set: {len(train_files)} images")
        print(f"Validation set: {len(val_files)} images")
        
        # Process train set (ORIGINAL DATA ONLY)
        self.process_split(train_files, 'train')
        
        # Process validation set
        self.process_split(val_files, 'val')
        
        # Create dataset yaml file
        self.create_dataset_yaml()
        
        print("Dataset preprocessing completed!")
    
    def process_split(self, files, split_name):
        """Process a specific split (train/val) - PRESERVES ORIGINAL DATA"""
        
        print(f"\nProcessing {split_name} split...")
        
        for i, (img_path, class_name) in enumerate(files):
            try:
                processed_img, scale, x_offset, y_offset = self.preprocess_image(img_path)
                
                # Generate unique filename
                original_filename = os.path.basename(img_path)
                img_filename = f"{class_name}_{i:04d}_{original_filename}"
                
                # Save processed image
                img_output_path = os.path.join(self.output_dir, split_name, 'images', img_filename)
                cv2.imwrite(img_output_path, processed_img)
                
                # Create and save annotation
                annotation = self.create_yolo_annotation(class_name, self.img_size, self.img_size)
                
                # Save label file
                label_filename = img_filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                label_output_path = os.path.normpath(os.path.join(self.output_dir, split_name, 'labels', label_filename))
                
                with open(label_output_path, 'w') as f:
                    f.write(annotation)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(files)} images for {split_name}")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"\nCompleted processing {split_name} split")
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        
        dataset_config = {
            'path': os.path.abspath(self.output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.classes),
            'names': self.classes
        }

        yaml_path = os.path.normpath(os.path.join(self.output_dir, 'dataset.yaml'))
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to {yaml_path}")
        return yaml_path

def main():
    print("\nStarting Sign Language Dataset Preprocessing for YOLO...")
    
    # Initialize preprocessor
    preprocessor = SignLanguageYOLOPreprocessor(
        source_dir="../datasets/final/train",
        output_dir="./yolo_dataset4",
        img_size=640
    )
    
    # Process the dataset
    preprocessor.process_dataset(train_ratio=0.8, val_ratio=0.2)
    
    # Print summary
    print("\nPreprocessing Summary:")
    print("=" * 50)
    print(f"✓ Created YOLO format dataset in './yolo_dataset4'")
    print(f"✓ Images resized to 640×640 pixels")
    print(f"✓ Train/Validation split: 80%/20%")
    print(f"✓ {len(preprocessor.classes)} classes: {preprocessor.classes}")
    print(f"✓ Dataset configuration: './yolo_dataset4/dataset.yaml'")

if __name__ == "__main__":
    main()

    dataset_path = "./yolo_dataset4/train/images"

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']

    fig, axes = plt.subplots(6, 6, figsize=(18, 18))
    fig.suptitle('Sample Preprocessed Images from Each Class (Original Data)', fontsize=16)

    for idx, class_name in enumerate(classes):
        row = idx // 6
        col = idx % 6
        
        class_images = [f for f in os.listdir(dataset_path) if f.startswith(class_name + "_")]
        
        if class_images:
            img_path = os.path.join(dataset_path, class_images[4])  # show the fifth one
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f'Class: {class_name}')
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error\n{class_name}', 
                                    ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')

    plt.tight_layout()
    # plt.savefig('./images/preprocessed_samples.png', dpi=300, bbox_inches='tight')
    # print("Sample preprocessed images saved as './images/preprocessed_samples.png'")
    plt.show()