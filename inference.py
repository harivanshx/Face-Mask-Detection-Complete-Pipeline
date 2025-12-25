import tensorflow as tf
import numpy as np
import cv2
import os

IMG_SIZE = 224
MODEL_PATH = ".\saved_model\mask_detector.keras"

CLASSES = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect"
}

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")


def preprocess_image(image_path):
    """
    Reads image from disk and prepares it for model inference
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_image = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return original_image, image


def predict(image_path):
    """
    Runs inference on a single image
    """
    original_image, input_image = preprocess_image(image_path)

    bbox_pred, class_pred = model.predict(input_image, verbose=0)

    bbox = bbox_pred[0]  # [xmin, ymin, xmax, ymax]
    class_id = np.argmax(class_pred[0])
    confidence = np.max(class_pred[0])

    label = CLASSES[class_id]

    return original_image, bbox, label, confidence


def draw_prediction(image, bbox, label, confidence):
    """
    Draws bounding box and label on image
    """
    h, w, _ = image.shape

    xmin = int(bbox[0] * w)
    ymin = int(bbox[1] * h)
    xmax = int(bbox[2] * w)
    ymax = int(bbox[3] * h)

    if label == "with_mask":
        color = (0, 255, 0)      # Green
    elif label == "without_mask":
        color = (0, 0, 255)      # Red
    else:
        color = (0, 255, 255)    # Yellow

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    text = f"{label} ({confidence:.2f})"
    cv2.putText(
        image,
        text,
        (xmin, max(ymin - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return image


if __name__ == "__main__":
    # Change this image path to test different images
    test_image_path = "./data/images/maksssksksss2.png"

    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Image not found: {test_image_path}")

    image, bbox, label, confidence = predict(test_image_path)
    result = draw_prediction(image, bbox, label, confidence)

    cv2.imshow("Face Mask Detection - Inference", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
