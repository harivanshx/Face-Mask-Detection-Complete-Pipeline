"""
Evaluation Script for Face Mask Detection Model
Run after training to get detailed metrics on test set
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_SAVE_PATH, CLASS_NAMES, IMG_SIZE, NUM_CLASSES
from src.dataset import build_datasets
from src.model import build_model


def evaluate_model():
    """Evaluate model on test set and generate metrics."""
    
    print("=" * 60)
    print("📊 Face Mask Detection - Model Evaluation")
    print("=" * 60)
    
    # Load datasets
    print("\n📂 Loading test data...")
    _, _, test_ds, _, _, n_test = build_datasets()
    print(f"   Test samples: {n_test}")
    
    # Load model
    print("\n🔧 Loading model...")
    try:
        model = build_model()
        model.load_weights(MODEL_SAVE_PATH)
        print(f"   Loaded weights from: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"   Error loading weights: {e}")
        print("   Trying to load full model...")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    # Evaluate
    print("\n📈 Running evaluation...")
    print("-" * 60)
    
    results = model.evaluate(test_ds, verbose=1)
    
    print("\n📊 Overall Results:")
    for name, value in zip(model.metrics_names, results):
        print(f"   {name}: {value:.4f}")
    
    # Get predictions for detailed metrics
    print("\n🔍 Generating detailed metrics...")
    
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        bbox_pred, class_pred = model.predict(images, verbose=0)
        
        true_labels = np.argmax(labels["class"].numpy(), axis=1)
        pred_labels = np.argmax(class_pred, axis=1)
        
        y_true.extend(true_labels)
        y_pred.extend(pred_labels)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification Report
    print("\n📋 Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion Matrix
    print("\n🔲 Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Calculate per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            acc = (y_pred[class_mask] == i).mean() * 100
            print(f"   {class_name}: {acc:.1f}%")
    
    # Save confusion matrix plot
    os.makedirs("evaluation_results", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha='right')
    plt.yticks(tick_marks, CLASS_NAMES)
    
    # Add values to cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix.png', dpi=150)
    print("\n💾 Confusion matrix saved to: evaluation_results/confusion_matrix.png")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    
    return results, y_true, y_pred


if __name__ == "__main__":
    evaluate_model()
