# Face Mask Detection - Training Workflow

This workflow describes how to train the face mask detection model from scratch.

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data is in the correct structure:
```
data/
├── images/          # PNG/JPG images
└── annotations/     # Pascal VOC XML files (same names as images)
```

## Training Pipeline

### Step 1: Configure Training (Optional)

Edit `src/config.py` to adjust:
- `BATCH_SIZE` - Default: 32
- `EPOCHS` - Default: 50
- `LEARNING_RATE` - Default: 1e-4
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` - Default: 70/15/15

### Step 2: Run Training

```bash
python pipeline.py
```

This will:
1. Split data into train/validation/test sets (70/15/15)
2. Apply data augmentation on training data
3. Train the model with early stopping
4. Save checkpoints to `checkpoints/`
5. Save final model to `saved_model/mask_detector.keras`
6. Log training to TensorBoard in `logs/`

### Step 3: Monitor Training (Optional)

In a separate terminal:
```bash
tensorboard --logdir logs
```

### Step 4: Evaluate Model

After training completes:
```bash
python evaluate.py
```

This generates:
- Classification report with precision/recall/F1
- Confusion matrix saved to `evaluation_results/`

### Step 5: Run the App

```bash
streamlit run app.py
```

## Output Files

- `saved_model/mask_detector.keras` - Final trained model
- `checkpoints/best_model.keras` - Best checkpoint during training
- `logs/` - TensorBoard logs
- `evaluation_results/confusion_matrix.png` - Confusion matrix visualization
