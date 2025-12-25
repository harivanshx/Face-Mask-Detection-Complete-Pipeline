"""
Dataset utilities for Face Mask Detection Pipeline
Handles data loading, augmentation, and train/val/test splitting
"""

import tensorflow as tf
import xml.etree.ElementTree as ET
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    IMG_SIZE, NUM_CLASSES, CLASSES, IMAGE_DIR, ANNOTATION_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, BATCH_SIZE
)


def parse_xml(xml_path):
    """
    Parse Pascal VOC XML annotation file.
    Returns the largest face's bounding box and label.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    boxes, labels = [], []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in CLASSES:
            continue
            
        label = CLASSES[class_name]
        box = obj.find("bndbox")

        xmin = int(box.find("xmin").text) / width
        ymin = int(box.find("ymin").text) / height
        xmax = int(box.find("xmax").text) / width
        ymax = int(box.find("ymax").text) / height

        # Clamp values to [0, 1]
        xmin = max(0, min(1, xmin))
        ymin = max(0, min(1, ymin))
        xmax = max(0, min(1, xmax))
        ymax = max(0, min(1, ymax))

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    if not boxes:
        return [0.25, 0.25, 0.75, 0.75], 0  # Default fallback

    # Choose largest face
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    idx = areas.index(max(areas))

    return boxes[idx], labels[idx]


def load_and_preprocess_image(img_path, augment=False):
    """Load and preprocess a single image."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    
    if augment:
        # Random augmentations for training
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    
    img = img / 255.0
    return img


def tf_data_generator(img_path, xml_path, augment=False):
    """TensorFlow-compatible data generator."""
    def _parse(img_p, xml_p):
        img = load_and_preprocess_image(img_p, augment)
        box, label = parse_xml(xml_p.numpy().decode("utf-8"))
        return img, np.array(box, dtype=np.float32), tf.one_hot(label, NUM_CLASSES)

    img, box, label = tf.py_function(
        _parse,
        [img_path, xml_path],
        [tf.float32, tf.float32, tf.float32]
    )

    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    box.set_shape((4,))
    label.set_shape((NUM_CLASSES,))

    return img, {"bbox": box, "class": label}


def get_file_pairs():
    """Get matching image and annotation file pairs."""
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    pairs = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        xml_file = base_name + ".xml"
        xml_path = os.path.join(ANNOTATION_DIR, xml_file)
        
        if os.path.exists(xml_path):
            img_path = os.path.join(IMAGE_DIR, img_file)
            pairs.append((img_path, xml_path))
    
    return pairs


def split_data(pairs):
    """
    Split data into train, validation, and test sets.
    Returns three lists of (image_path, xml_path) tuples.
    """
    np.random.seed(RANDOM_SEED)
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        pairs, 
        test_size=TEST_RATIO, 
        random_state=RANDOM_SEED
    )
    
    # Second split: train vs val
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val, 
        test_size=val_size_adjusted, 
        random_state=RANDOM_SEED
    )
    
    print(f"📊 Data Split:")
    print(f"   Train: {len(train)} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"   Val:   {len(val)} samples ({VAL_RATIO*100:.0f}%)")
    print(f"   Test:  {len(test)} samples ({TEST_RATIO*100:.0f}%)")
    
    return train, val, test


def create_dataset(pairs, batch_size=BATCH_SIZE, augment=False, shuffle=True):
    """
    Create a tf.data.Dataset from file pairs.
    """
    if not pairs:
        raise ValueError("No data pairs provided")
    
    image_paths = [p[0] for p in pairs]
    xml_paths = [p[1] for p in pairs]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, xml_paths))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(pairs), seed=RANDOM_SEED)
    
    # Map with augmentation flag
    map_fn = lambda img, xml: tf_data_generator(img, xml, augment)
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def build_datasets():
    """
    Main function to build train, validation, and test datasets.
    Returns three tf.data.Dataset objects.
    """
    print("🔍 Loading file pairs...")
    pairs = get_file_pairs()
    print(f"   Found {len(pairs)} image-annotation pairs")
    
    if len(pairs) == 0:
        raise ValueError("No valid image-annotation pairs found!")
    
    print("\n✂️ Splitting data...")
    train_pairs, val_pairs, test_pairs = split_data(pairs)
    
    print("\n📦 Creating datasets...")
    train_ds = create_dataset(train_pairs, augment=True, shuffle=True)
    val_ds = create_dataset(val_pairs, augment=False, shuffle=False)
    test_ds = create_dataset(test_pairs, augment=False, shuffle=False)
    
    return train_ds, val_ds, test_ds, len(train_pairs), len(val_pairs), len(test_pairs)


# For backward compatibility
def build_dataset(image_dir, xml_dir, batch_size=32):
    """Legacy function - builds a single dataset without splits."""
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    xml_paths = sorted([os.path.join(xml_dir, f) for f in os.listdir(xml_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, xml_paths))
    
    map_fn = lambda img, xml: tf_data_generator(img, xml, augment=False)
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
