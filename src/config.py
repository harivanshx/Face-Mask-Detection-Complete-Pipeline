"""
Configuration file for Face Mask Detection Pipeline
"""

# Data paths
DATA_DIR = "./data"
IMAGE_DIR = "./data/images"
ANNOTATION_DIR = "./data/annotations"
MODEL_SAVE_PATH = "./saved_model/mask_detector.keras"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# Model configuration
IMG_SIZE = 224
NUM_CLASSES = 3
BACKBONE = "MobileNetV2"

# Class mapping
CLASSES = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Data augmentation settings
AUGMENTATION = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "zoom_range": 0.1
}
