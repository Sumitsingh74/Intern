import os
# If you had torch-related watcher issues, you can disable file watcher early:
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import torch
if hasattr(torch, "classes"):
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

import cv2
import streamlit as st
from pathlib import Path
from utils import load_model
import config

# # Streamlit page config
# st.set_page_config(
#     page_title="YOLOv8 Webcam Detection",
#     page_icon="🎥",
#     layout="wide"
# )
st.title("🔍 Webcam Detection (Click to capture frames)")

# Sidebar: model settings
st.sidebar.header("Model Config")
model_type = st.sidebar.selectbox("Select Model", config.DETECTION_MODEL_LIST)
confidence = st.sidebar.slider("Confidence (%)", 30, 100, 50) / 100.0

# Load model once
@st.cache_resource
def load_yolo_model(path: Path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"❌ Failed to load model at {path}: {e}")
        return None

model_path = Path(config.DETECTION_MODEL_DIR) / model_type
model = load_yolo_model(model_path)
if model is None:
    st.stop()

# Session state for webcam and run flag
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None

# Start/Stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Webcam"):
        if not st.session_state.run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access webcam.")
            else:
                st.session_state.cap = cap
                st.session_state.run = True
with col2:
    if st.button("⏹️ Stop Webcam"):
        if st.session_state.run:
            cap = st.session_state.cap
            if cap is not None:
                cap.release()
            st.session_state.cap = None
            st.session_state.run = False

# Placeholders for display
original_placeholder = st.empty()
annotated_placeholder = st.empty()
alert_placeholder = st.empty()
import time
# If running, show a button to capture the next frame
if st.session_state.run:
    while(True):
        time.sleep(1)
        cap = st.session_state.cap
        if cap is None:
            st.error("❌ Webcam not initialized.")
            st.session_state.run = False
        else:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read frame from webcam.")
                cap.release()
                st.session_state.cap = None
                st.session_state.run = False
            else:
                # Run YOLOv8 detection
                try:
                    results = model(frame, conf=confidence)
                    annotated_frame = results[0].plot()
                except Exception as e:
                    st.error(f"❌ Inference error: {e}")
                    annotated_frame = frame.copy()

                # Count detections
                names = results[0].names if hasattr(results[0], "names") else {}
                boxes = results[0].boxes if results else None
                classes = boxes.cls.tolist() if boxes is not None else []
                labels = [names[int(c)] for c in classes]

                person_count = labels.count("person")
                phone_count = labels.count("cell phone")

                # Show alert
                if person_count == 0:
                    alert_placeholder.error("🚨 No person detected!")
                elif person_count > 1:
                    alert_placeholder.warning(f"⚠️ Multiple people detected: {person_count}")
                elif phone_count > 0:
                    alert_placeholder.info(f"📱 Cell phone detected: {phone_count}")
                else:
                    alert_placeholder.success("✅ Normal: 1 person, no phone detected.")

                # *** Only show the annotated frame, small like a webcam ***
                annotated_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    caption="Detection",
                    use_column_width=False,
                    width=320  # adjust to taste
                )
else:
    # If not running, clear placeholders and release if needed
    original_placeholder.empty()
    annotated_placeholder.empty()
    alert_placeholder.empty()
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
