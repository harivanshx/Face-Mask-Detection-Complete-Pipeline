"""
Complete Training Pipeline for Face Mask Detection
Run this script to train the model from scratch with proper splits
"""

import os
import sys
import tensorflow as tf
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, CHECKPOINT_DIR, LOG_DIR,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    BATCH_SIZE, CLASS_NAMES
)
from src.dataset import build_datasets
from src.model import build_model


def create_callbacks():
    """Create training callbacks."""
    
    # Ensure directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, timestamp),
            histogram_freq=1
        ),
        
        # Progress bar
        tf.keras.callbacks.ProgbarLogger()
    ]
    
    return callbacks


def compile_model(model):
    """Compile the model with loss functions and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "bbox": "mse",
            "class": "categorical_crossentropy"
        },
        loss_weights={
            "bbox": 1.0,
            "class": 1.0
        },
        metrics={
            "class": ["accuracy"]
        }
    )
    return model


def train():
    """Main training function."""
    print("=" * 60)
    print("🎭 Face Mask Detection - Training Pipeline")
    print("=" * 60)
    
    # Build datasets
    print("\n📂 Step 1: Loading and splitting data...")
    train_ds, val_ds, test_ds, n_train, n_val, n_test = build_datasets()
    
    # Build model
    print("\n🏗️ Step 2: Building model...")
    model = build_model()
    model = compile_model(model)
    model.summary()
    
    # Create callbacks
    print("\n⚙️ Step 3: Setting up callbacks...")
    callbacks = create_callbacks()
    
    # Calculate steps
    steps_per_epoch = n_train // BATCH_SIZE
    validation_steps = max(1, n_val // BATCH_SIZE)
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Steps per Epoch: {steps_per_epoch}")
    print(f"   Validation Steps: {validation_steps}")
    
    # Train
    print("\n🚀 Step 4: Starting training...")
    print("-" * 60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n💾 Step 5: Saving model...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"   Model saved to: {MODEL_SAVE_PATH}")
    
    # Evaluate on test set
    print("\n📈 Step 6: Evaluating on test set...")
    print("-" * 60)
    
    test_results = model.evaluate(test_ds, verbose=1)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📊 Test Results:")
    for name, value in zip(model.metrics_names, test_results):
        print(f"   {name}: {value:.4f}")
    
    # Return history for analysis
    return history, model


if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🖥️ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")
    
    # Run training
    history, model = train()
