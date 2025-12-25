import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from src.model import build_model
from mtcnn import MTCNN

# Page config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# Constants
IMG_SIZE = 224
MODEL_PATH = "saved_model/mask_detector.keras"
CLASSES = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect"
}

COLORS = {
    "with_mask": (0, 255, 0),        # Green
    "without_mask": (255, 0, 0),      # Red
    "mask_weared_incorrect": (255, 165, 0)  # Orange
}

@st.cache_resource
def load_mask_classifier():
    """Rebuilds the architecture and loads weights for mask classification."""
    try:
        model = build_model()
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading mask classifier: {e}")
        return None

@st.cache_resource
def load_face_detector():
    """Loads MTCNN face detector."""
    try:
        detector = MTCNN()
        return detector
    except Exception as e:
        st.error(f"Error loading face detector: {e}")
        return None

def classify_face(face_crop, model):
    """Classifies a single face crop for mask status."""
    # Resize to model input size
    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    face_normalized = face_resized / 255.0
    
    # Expand dims for batch
    input_image = np.expand_dims(face_normalized, axis=0)
    
    # Predict - model outputs [bbox, class], we only need class
    _, class_pred = model.predict(input_image, verbose=0)
    
    class_id = np.argmax(class_pred[0])
    confidence = float(np.max(class_pred[0]))
    label = CLASSES[class_id]
    
    return label, confidence

def detect_and_classify(image, face_detector, mask_classifier):
    """
    1. Detects all faces using MTCNN
    2. Classifies each face for mask status
    3. Returns annotated image and results
    """
    # Convert PIL to numpy RGB
    img_array = np.array(image.convert('RGB'))
    annotated_image = img_array.copy()
    
    # Detect faces with MTCNN
    detections = face_detector.detect_faces(img_array)
    
    results = []
    
    if len(detections) == 0:
        return annotated_image, [], "No faces detected in the image."
    
    for detection in detections:
        try:
            # Get bounding box
            x, y, w, h = detection['box']
            
            # Ensure coordinates are valid
            x = max(0, x)
            y = max(0, y)
            
            # Crop face
            face_crop = img_array[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                continue
            
            # Classify
            label, confidence = classify_face(face_crop, mask_classifier)
            
            # Get color
            color = COLORS.get(label, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 3)
            
            # Draw label
            text = f"{label} ({confidence:.0%})"
            text_y = y - 10 if y - 10 > 20 else y + h + 25
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated_image, (x, text_y - text_h - 5), (x + text_w + 5, text_y + 5), color, -1)
            
            cv2.putText(
                annotated_image,
                text,
                (x + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # White text
                2
            )
            
            results.append({
                "label": label,
                "confidence": confidence,
                "box": (x, y, w, h)
            })
            
        except Exception as e:
            print(f"Error processing face: {e}")
            continue
    
    return annotated_image, results, None

def main():
    st.title("😷 Face Mask Detection")
    st.markdown("""
    Upload an image to detect faces and check mask status.
    
    **Classes:**
    - 🟢 **With Mask** - Properly wearing a mask
    - 🔴 **Without Mask** - Not wearing a mask  
    - 🟠 **Incorrect** - Mask worn incorrectly
    """)
    
    # Load models
    mask_classifier = load_mask_classifier()
    face_detector = load_face_detector()
    
    if mask_classifier is None or face_detector is None:
        st.error("Failed to load models. Please check the console for errors.")
        return

    st.success("✅ Models loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a photo with one or more faces"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, width=300)
        
        if st.button("🔍 Detect Masks", type="primary"):
            with st.spinner('Detecting faces and analyzing masks...'):
                result_image, results, error = detect_and_classify(
                    image, face_detector, mask_classifier
                )
                
                with col2:
                    st.subheader("Result")
                    st.image(result_image, width=300)
                
                if error:
                    st.warning(error)
                elif results:
                    st.success(f"✅ Found {len(results)} face(s)")
                    
                    # Summary
                    with_mask = sum(1 for r in results if r['label'] == 'with_mask')
                    without_mask = sum(1 for r in results if r['label'] == 'without_mask')
                    incorrect = sum(1 for r in results if r['label'] == 'mask_weared_incorrect')
                    
                    cols = st.columns(3)
                    cols[0].metric("🟢 With Mask", with_mask)
                    cols[1].metric("🔴 Without Mask", without_mask)
                    cols[2].metric("🟠 Incorrect", incorrect)

if __name__ == "__main__":
    main()
