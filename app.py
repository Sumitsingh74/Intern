import os
# (optional) work around torch watcher issues
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import torch
if hasattr(torch, "classes"):
    torch.classes.__path__ = []

import cv2
import streamlit as st
from pathlib import Path
from utils import load_model
import config
import time

# NEW: auto–refresh the app every 1 second
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=1000, key="auto")

st.title("🔍 Webcam Detection (auto–refresh)")

# Sidebar
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

model = load_yolo_model(Path(config.DETECTION_MODEL_DIR) / model_type)
if model is None:
    st.stop()

# Session state
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None

# If somehow cap is still open but run=False (e.g. after a refresh or closing the tab),
# we immediately release it so we don’t leave the camera locked.
if not st.session_state.run and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None

# Start/Stop
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
            st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.run = False

st.markdown("---")

# Placeholders
alert_placeholder = st.empty()
annotated_placeholder = st.empty()

# One frame per run
if st.session_state.run:
    cap = st.session_state.cap
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Failed to read frame from webcam.")
        # If capture ever fails, clean up
        cap.release()
        st.session_state.cap = None
        st.session_state.run = False
    else:
        # YOLOv8 inference
        try:
            results = model(frame, conf=confidence)
            annotated = results[0].plot()
        except Exception as e:
            st.error(f"❌ Inference error: {e}")
            annotated = frame.copy()

        # Count detections
        names = results[0].names
        cls = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
        labels = [names[int(c)] for c in cls]
        p_count = labels.count("person")
        ph_count = labels.count("cell phone")

        # Alert
        if p_count == 0:
            alert_placeholder.error("🚨 No person detected!")
        elif p_count > 1:
            alert_placeholder.warning(f"⚠️ Multiple people: {p_count}")
        elif ph_count > 0:
            alert_placeholder.info(f"📱 Cell phone detected: {ph_count}")
        else:
            alert_placeholder.success("✅ 1 person, no phone.")

        # *** Only show annotated frame, small like a webcam ***
        annotated_placeholder.image(
            annotated,
            channels="BGR",
            caption="Detection",
            use_column_width=False,
            width=320
        )
else:
    st.info("Click ▶️ Start to begin automatic detection.")
