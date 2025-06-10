
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


import time  # Add this import at the top

import time

def infer_uploaded_webcam(conf, model):
    """
    Run YOLOv8 inference on webcam feed with Start/Stop controls.
    Show alerts based on person and cell phone count.
    """

    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    # Start/Stop Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Webcam"):
            st.session_state.run_webcam = True
    with col2:
        if st.button("⏹️ Stop Webcam"):
            st.session_state.run_webcam = False

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Failed to open webcam.")
            return

        # Display placeholders
        col1, col2 = st.columns(2)
        orig_placeholder = col1.empty()
        detect_placeholder = col2.empty()

        # Alert area
        alert_placeholder = st.empty()

        while cap.isOpened() and st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Couldn't read from webcam.")
                break

            results = model(frame, conf=conf)
            annotated_frame = results[0].plot()

            # Count detections
            names = results[0].names
            boxes = results[0].boxes
            classes = boxes.cls.tolist() if boxes is not None else []
            labels = [names[int(c)] for c in classes]

            person_count = labels.count("person")
            phone_count = labels.count("cell phone")

            # Alerts based on detection
            if person_count == 0:
                alert_placeholder.error("🚨 No person detected!")
            elif person_count > 1:
                alert_placeholder.warning(f"⚠️ Multiple people detected: {person_count}")
            elif phone_count > 0:
                alert_placeholder.info(f"📱 Cell phone detected: {phone_count}")

            # Show images
            orig_placeholder.image(frame, channels="BGR", caption="Original", use_column_width=True)
            detect_placeholder.image(annotated_frame, channels="BGR", caption="Detection", use_column_width=True)

            time.sleep(3)

        cap.release()
        cv2.destroyAllWindows()
