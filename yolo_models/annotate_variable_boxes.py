"""Create a manually verified YOLO dataset with variable hand boxes.

The source data is arranged as ``source/class_name/image``. One image is
shown at a time; drag a rectangle around the hand and press Enter or Space
to save it. Press R to redraw, S to skip, or Q to stop.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import yaml


CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../datasets/final/train"))
    parser.add_argument("--output", type=Path, default=Path("./yolo_variable_boxes"))
    parser.add_argument("--sample-per-class", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="List the sample without opening annotation windows")
    return parser.parse_args()


def collect_sample(source, sample_per_class, seed):
    rng = random.Random(seed)
    sampled = []
    for class_name in CLASSES:
        class_dir = source / class_name
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if class_dir.is_dir() else []
        if len(images) < sample_per_class:
            raise ValueError(
                f"Class {class_name!r} has {len(images)} images; "
                f"cannot sample {sample_per_class}."
            )
        sampled.extend((class_name, path) for path in rng.sample(images, sample_per_class))
    return sampled


def yolo_label(class_index, box, image_width, image_height):
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2) / image_width
    center_y = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"{class_index} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def annotate_image(image, window_name):
    display = image.copy()
    box = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, width, height = map(int, box)
    if width == 0 or height == 0:
        return None
    return x, y, x + width, y + height


def main():
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    sampled = collect_sample(args.source, args.sample_per_class, args.seed)
    print(f"Selected {len(sampled)} images ({args.sample_per_class} per class).")

    if args.dry_run:
        for class_name, image_path in sampled:
            print(f"{class_name}\t{image_path}")
        return

    split_samples = {}
    for class_name in CLASSES:
        class_samples = [item for item in sampled if item[0] == class_name]
        validation_count = max(1, round(len(class_samples) * args.val_ratio))
        validation_paths = {path for _, path in class_samples[:validation_count]}
        for item in class_samples:
            split_samples[item[1]] = "val" if item[1] in validation_paths else "train"

    for split in ("train", "val"):
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "annotations.csv"
    class_to_index = {name: index for index, name in enumerate(CLASSES)}

    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        metadata = csv.writer(metadata_file)
        metadata.writerow(["split", "class_name", "class_id", "source", "image", "x1", "y1", "x2", "y2"])

        for position, (class_name, image_path) in enumerate(sampled, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            height, width = image.shape[:2]
            window_name = f"Annotating {position}/{len(sampled)}: {class_name}"
            print(f"[{position}/{len(sampled)}] {image_path}")
            print("Drag around the hand, then press Enter/Space to save; R to redraw; S to skip; Q to stop.")

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("Stopped by user. Existing annotations were retained.")
                    return
                box = annotate_image(image, window_name)
                if box is None:
                    print("No box selected; press R to retry, S to skip, or Q to stop.")
                    choice = cv2.waitKey(0) & 0xFF
                    if choice in (ord("q"), ord("Q")):
                        return
                    if choice in (ord("s"), ord("S")):
                        break
                    continue
                break

            if box is None:
                continue

            split = split_samples[image_path]
            output_name = f"{class_name}_{position:04d}_{image_path.name}"
            output_image = args.output / split / "images" / output_name
            output_label = args.output / split / "labels" / f"{Path(output_name).stem}.txt"
            shutil.copy2(image_path, output_image)
            output_label.write_text(
                yolo_label(class_to_index[class_name], box, width, height) + "\n",
                encoding="utf-8",
            )
            metadata.writerow([split, class_name, class_to_index[class_name], str(image_path), output_name, *box])
            metadata_file.flush()

    yaml_path = args.output / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump({
            "path": str(args.output.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(CLASSES),
            "names": CLASSES,
        }, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote annotations to {args.output}")


if __name__ == "__main__":
    main()