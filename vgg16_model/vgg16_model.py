import os
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
import time
warnings.filterwarnings('ignore')
import pickle
import csv
import statistics

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SignLanguageVGG16:
    def __init__(self, data_path='../datasets/final/train', img_size=(224, 224), batch_size=32):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = []
        
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for saving models and images"""
        os.makedirs('./models_non_augm_data', exist_ok=True)
        os.makedirs('./images_non_augm_data', exist_ok=True)

    def create_data_generators(self, use_augmentation=True, seed=None):
        """Create data generators with optional training augmentation"""
        print("=" * 60)
        print("CREATING DATA GENERATORS")
        print("=" * 60)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist!")
        
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=False,
                fill_mode='nearest',
                validation_split=0.2
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=seed
        )

        self.class_names = list(train_generator.class_indices.keys())
        
        validation_generator = validation_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=seed
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Number of classes: {train_generator.num_classes}")

        self.class_names = list(train_generator.class_indices.keys())
        print(f"Classes: {self.class_names}")
        return train_generator, validation_generator

    def build_vgg16_model(self):
        """Build the VGG16 model with custom top layers"""
        print("=" * 60)
        print("BUILDING VGG16 MODEL")
        print("=" * 60)
        
        if not self.class_names:
            raise ValueError("No classes found. Create data generators first!")
            
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        base_model.trainable = False

        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("\nModel Summary:")
        self.model.summary()
        print("=" * 60)
        return model

    def train_model(self, train_generator, validation_generator, epochs=10,
                    result_directory='./models_non_augm_data'):
        """Train the VGG16 model"""
        print("=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Model not built. Call build_vgg16_model() first!")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(result_directory, 'best_vgg16_model_10-32.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Train model
        print(f"Training for {epochs} epochs...")
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        with open(os.path.join(result_directory, 'vgg16_training_history_10-32.pkl'), 'wb') as f:
            pickle.dump(self.history.history, f)

        # get accuracy and loss
        train_accuracy = self.history.history['accuracy'][-1]
        val_accuracy = self.history.history['val_accuracy'][-1]
        train_loss = self.history.history['loss'][-1]
        val_loss = self.history.history['val_loss'][-1]
        print(f"\nFinal Training Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")
        print(f"Final Validation Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")

        print("Training completed!")
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./images_non_augm_data/vgg16_training_history_10-32.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, validation_generator):
        """Evaluate model performance"""
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Model not trained. Train the model first!")
        
        # Get predictions
        validation_generator.reset()
        predictions = self.model.predict(validation_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = validation_generator.classes
        class_labels = list(validation_generator.class_indices.keys())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=class_labels))
        
        accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        
        return accuracy

    # def save_model(self, filename='./models/vgg16_sign_language_model.keras'):
    #     """Save the trained model"""
    #     if self.model is None:
    #         print("No model to save!")
    #         return
            
    #     self.model.save(filename)
    #     print(f"Model saved as {filename}")

def main():
    """Main function to run the complete pipeline"""
    print("VGG16 Sign Language Detection Model")
    print("====================================")
    
    try:
        # seeds = [42, 17, 119]
        seeds = [17, 119]
        batch_sizes = [8, 16, 32]
        paradigms = [
            ('non_augmented', False, './models_non_augm_data'),
            ('augmented', True, './models_augm_data'),
        ]
        result_fields = [
            'paradigm',
            'seed',
            'train_accuracy',
            'validation_accuracy',
            'train_loss',
            'validation_loss',
            'training_time_seconds',
        ]
        metric_fields = result_fields[2:]

        for paradigm, use_augmentation, result_directory in paradigms:
            os.makedirs(result_directory, exist_ok=True)
            run_results = []
            print(f"\n===== {paradigm.upper()} EXPERIMENTS =====")

            for seed in seeds:
                for batch_size in batch_sizes:
                    print(
                        f"\nStarting {paradigm} VGG16 run with seed {seed}, "
                        f"batch size {batch_size}..."
                    )
                    np.random.seed(seed)
                    tf.random.set_seed(seed)

                    vgg16_model = SignLanguageVGG16(
                        data_path='../datasets/final/train',
                        img_size=(224, 224),
                        batch_size=batch_size
                    )

                    train_gen, val_gen = vgg16_model.create_data_generators(
                        use_augmentation=use_augmentation,
                        seed=seed
                    )
                    model = vgg16_model.build_vgg16_model()

                    run_directory = os.path.join(
                        result_directory,
                        f'batch_{batch_size}',
                        f'seed_{seed}'
                    )
                    os.makedirs(run_directory, exist_ok=True)
                    start_time = time.time()
                    history = vgg16_model.train_model(
                        train_gen,
                        val_gen,
                        epochs=10,
                        result_directory=run_directory
                    )
                    training_time = time.time() - start_time
                    evaluated_accuracy = vgg16_model.evaluate_model(val_gen)

                    model.save(os.path.join(run_directory, 'vgg16_model.keras'))
                    with open(os.path.join(run_directory, 'training_history.pkl'), 'wb') as history_file:
                        pickle.dump(history.history, history_file)

                    run_result = {
                        'paradigm': paradigm,
                        'seed': seed,
                        'batch_size': batch_size,
                        'train_accuracy': history.history['accuracy'][-1],
                        'validation_accuracy': evaluated_accuracy,
                        'train_loss': history.history['loss'][-1],
                        'validation_loss': history.history['val_loss'][-1],
                        'training_time_seconds': training_time,
                    }
                    run_results.append(run_result)
                    print(
                        f"Run seed={seed}, batch={batch_size}: "
                        f"validation accuracy={evaluated_accuracy:.4f}, "
                        f"training time={training_time:.2f}s"
                    )
                    tf.keras.backend.clear_session()

            with open(os.path.join(result_directory, 'vgg16_results.csv'), 'w', newline='') as results_file:
                writer = csv.DictWriter(
                    results_file,
                    fieldnames=['paradigm', 'seed', 'batch_size', *metric_fields]
                )
                writer.writeheader()
                writer.writerows(run_results)

            with open(os.path.join(result_directory, 'vgg16_summary.csv'), 'w', newline='') as summary_file:
                writer = csv.DictWriter(
                    summary_file,
                    fieldnames=['group', 'metric', 'mean', 'sample_standard_deviation', 'runs']
                )
                writer.writeheader()
                summary_groups = [('All runs', run_results)]
                summary_groups.extend(
                    (
                        f'Batch size {batch_size}',
                        [result for result in run_results if result['batch_size'] == batch_size]
                    )
                    for batch_size in batch_sizes
                )
                for group, group_results in summary_groups:
                    print(f"\n{paradigm} {group} summary (mean +/- sample standard deviation):")
                    for field in metric_fields:
                        values = [result[field] for result in group_results]
                        mean = statistics.mean(values)
                        standard_deviation = statistics.stdev(values)
                        writer.writerow({
                            'group': group,
                            'metric': field,
                            'mean': mean,
                            'sample_standard_deviation': standard_deviation,
                            'runs': len(values),
                        })
                        print(f"{field}: {mean:.4f} +/- {standard_deviation:.4f}")

        print("Model training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your data path and directory structure.")

if __name__ == "__main__":
    main()