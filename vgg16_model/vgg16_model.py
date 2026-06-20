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

    def create_data_generators(self):
        """Create data generators with augmentation"""
        print("=" * 60)
        print("CREATING DATA GENERATORS")
        print("=" * 60)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist!")
        
        # Data augmentation for training
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
            shuffle=True
        )

        self.class_names = list(train_generator.class_indices.keys())
        
        validation_generator = validation_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
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

    def train_model(self, train_generator, validation_generator, epochs=10):
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
                './models_non_augm_data/best_vgg16_model_10-32.keras',
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
        with open('./models_non_augm_data/vgg16_training_history_10-32.pkl', 'wb') as f:
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
        # Initialize the model
        vgg16_model = SignLanguageVGG16(data_path='../datasets/final/train', img_size=(224, 224), batch_size=32)
        
        # Create data generators
        train_gen, val_gen = vgg16_model.create_data_generators()

        # Build model
        model = vgg16_model.build_vgg16_model()

        # Train model
        start_time = time.time()
        history = vgg16_model.train_model(train_gen, val_gen, epochs=10)
        training_time = time.time() - start_time
        minutes = int(training_time // 60)
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
        seconds = int(training_time % 60)
        print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")

        # Plot training history
        vgg16_model.plot_training_history()

        # Evaluate model
        # accuracy = vgg16_model.evaluate_model(val_gen)

        # print(f"\nFinal Validation Accuracy: {accuracy:.4f}")
        print("Model training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your data path and directory structure.")

if __name__ == "__main__":
    main()