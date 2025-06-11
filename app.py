import cv2
import streamlit as st
from PIL import Image
from pathlib import Path
from utils import load_model
import config

# Streamlit page config
st.set_page_config(
    page_title="YOLOv8 Webcam Detection",
    page_icon="🎥",
    layout="wide"
)

st.title("🔍 Live Webcam Detection with YOLOv8")

# Sidebar: model settings
st.sidebar.header("Model Config")
model_type = st.sidebar.selectbox(
    "Select Model",
    config.DETECTION_MODEL_LIST
)
confidence = float(st.sidebar.slider("Confidence", 30, 100, 50)) / 100

# Load model
model_path = Path(config.DETECTION_MODEL_DIR, model_type)
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Initialize session state
if "run" not in st.session_state:
    st.session_state.run = False

# Buttons: Start & Stop (side by side)
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start"):
        st.session_state.run = True
with col2:
    if st.button("⏹️ Stop"):
        st.session_state.run = False

# Only run webcam detection if started
import time
# import cv2
# import streamlit as st

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
        st.stop()

    time.sleep(3)

    # Create two columns for original and annotated video
    col1, col2 = st.columns(2)
    original_placeholder = col1.empty()
    annotated_placeholder = col2.empty()

    # Placeholder for alerts
    alert_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or not st.session_state.run:
            break

        # Run YOLOv8 detection
        results = model(frame, conf=confidence)
        annotated_frame = results[0].plot()

        # Count detections
        names = results[0].names
        boxes = results[0].boxes
        classes = boxes.cls.tolist() if boxes is not None else []
        labels = [names[int(c)] for c in classes]

        person_count = labels.count("person")
        phone_count = labels.count("cell phone")

        # Show alert on frontend
        if person_count == 0:
            alert_placeholder.error("🚨 No person detected!")
        elif person_count > 1:
            alert_placeholder.warning(f"⚠️ Multiple people detected: {person_count}")
        elif phone_count > 0:
            alert_placeholder.info(f"📱 Cell phone detected: {phone_count}")
        else:
            alert_placeholder.success("✅ Normal: 1 person, no phone detected.")

        # Display frames
        original_placeholder.image(frame, channels="BGR", caption="Original", use_column_width=True)
        annotated_placeholder.image(annotated_frame, channels="BGR", caption="Detection", use_column_width=True)

        time.sleep(2)

    cap.release()
    cv2.destroyAllWindows()
