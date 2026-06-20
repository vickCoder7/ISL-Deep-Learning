from email.mime import image
import os
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import random
from preprocessing import SignLanguageYOLOPreprocessor

yolo_preprocessor = SignLanguageYOLOPreprocessor()
np.random.seed(42)
random.seed(42)

def test_multiple_images():
    """Test model on 3 random images per class but visualize only 1 per class"""

    model_path = "./sign_language_detection/yolov8_sign_detection_split3(rs=119)/weights/best.pt"
    model = YOLO(model_path)

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']

    dataset_path = "../datasets/final2/train"

    fig, axes = plt.subplots(6, 6, figsize=(20, 20))
    fig.suptitle('YOLO Sign Language Detection - Sample Results (3 tests per class, 1 shown)',
                 fontsize=16, fontweight='bold')

    results_info = []

    for idx, class_name in enumerate(class_names):
        row = idx // 6
        col = idx % 6

        class_dir = os.path.join(dataset_path, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                # Pick up to 3 random test images
                test_images = random.sample(images, min(3, len(images)))

                visualized = False  # track whether we already displayed one image
                detection_info = "No Detection"

                for test_image in test_images:
                    image_path = os.path.join(class_dir, test_image)

                    # # Preprocess and predict
                    # image = cv2.imread(image_path)
                    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    preprocessed_image, scale, x_offset, y_offset = yolo_preprocessor.preprocess_image(image_path, target_size=640)
                    results = model.predict(preprocessed_image, verbose=False)

                    annotated_image = preprocessed_image.copy()

                    if results and len(results) > 0:
                        result = results[0]
                        if result.boxes is not None and len(result.boxes) > 0:
                            for i in range(len(result.boxes)):
                                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                                conf = result.boxes.conf[i].cpu().numpy()
                                cls = int(result.boxes.cls[i].cpu().numpy())

                                if conf > 0.5:
                                    # Annotate only if visualizing
                                    if not visualized:
                                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2),
                                                      (0, 255, 0), 3)
                                        label = f"{class_names[cls]}: {conf:.2f}"
                                        cv2.putText(annotated_image, label, (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                    (0, 255, 0), 2)

                                    detection_info = f"Pred: {class_names[cls]} ({conf:.3f})"
                                    is_correct = class_names[cls] == class_name
                                    results_info.append({
                                        'true': class_name,
                                        'pred': class_names[cls],
                                        'conf': conf,
                                        'correct': is_correct
                                    })

                    # Show only the first tested image per class
                    if not visualized:
                        axes[row, col].imshow(annotated_image)
                        axes[row, col].set_title(f"True: {class_name}\n{detection_info}",
                                                 fontsize=10)
                        axes[row, col].axis('off')
                        visualized = True

    plt.tight_layout()
    # plt.savefig('../vgg16_model/images/200img_prep_test_results_vgg16.png', dpi=300, bbox_inches='tight')
    # print("Visual test results saved at '../vgg16_model/images/200img_prep_test_results_vgg16.png'")

    # Print summary accuracy
    if results_info:
        correct_count = sum(1 for r in results_info if r['correct'])
        total_count = len(results_info)
        accuracy = correct_count / total_count * 100

        print(f"\n✅ TEST SUMMARY (3 per class):")
        print(f"\tImages tested: {total_count}")
        print(f"\tCorrect predictions: {correct_count}")
        print(f"\tAccuracy: {accuracy:.1f}%")
        
        # Show individual results
        print(f"\n📋 DETAILED RESULTS:")
        for i, r in enumerate(results_info, 1):
            status = "✅" if r['correct'] else "❌"
            print(f"   {i}. True: {r['true']}, Pred: {r['pred']}, Conf: {r['conf']:.3f} {status}")

def test_model_on_validation_set():
    """Test model on validation set images"""

    model_path = "./sign_language_detection/yolov8_sign_detection_split3(rs=119)/weights/best.pt"
    model = YOLO(model_path)
    
    val_dir = "./yolo_dataset/val/images"
    
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return
    
    print(f"\n🧪 Testing on validation set...")
    
    # Run validation
    results = model.val(data="./yolo_dataset/dataset.yaml", split='val', verbose=True)
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   mAP50: {results.box.map50:.4f}")
    print(f"   mAP50-95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")

def benchmark_speed():
        """Benchmark model inference speed"""

        model_path = "./sign_language_detection/yolov8_sign_detection_split3(rs=119)/weights/best.pt"
        model = YOLO(model_path)
        num_iterations=10

        print(f"\n⏱️  SPEED BENCHMARK ({num_iterations} iterations)")
        print("="*40)
        
        if model is None:
            print("❌ Model not loaded")
            return
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            model.predict(dummy_image, verbose=False)
        
        # Benchmark
        import time
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            results = model.predict(dummy_image, verbose=False)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(inference_time)
            print(f"  Iteration {i+1}: {inference_time:.2f}ms")
        
        # Statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time
        
        print(f"\n📊 SPEED STATISTICS:")
        print(f"   Average: {avg_time:.2f}ms ({fps:.1f} FPS)")
        print(f"   Min: {min_time:.2f}ms")
        print(f"   Max: {max_time:.2f}ms")

if __name__ == "__main__":
    print("=" * 50)
    print("🎨 YOLO Sign Language Visual Testing")
    print("=" * 50)
    
    # Test 1: Visual grid test
    test_multiple_images()
    
    # Test 2: Full validation set test
    # test_model_on_validation_set()

    # Test 3: Speed benchmark
    benchmark_speed()
    
    print("\n🎉 All tests completed!")