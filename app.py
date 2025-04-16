import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Detection Dashboard", layout="wide")
st.title("🎯 YOLOv11 Object Detection Demo")

# Sidebar controls for detection info
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
run_detection = st.sidebar.button("Start Detection")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Load logo if exists
logo_path = "logo.png"
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED) if os.path.exists(logo_path) else None

# Create a dict to store class counts for model labels
class_counts = {name: 0 for name in model.names}  # Initialized but not displayed

def overlay_logo(frame, logo, pos=(10, 10)):
    h_logo, w_logo = logo.shape[:2]
    x, y = pos
    if logo.shape[2] == 4:
        alpha_logo = logo[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h_logo, x:x+w_logo, c] = (
                alpha_logo * logo[:, :, c] +
                (1 - alpha_logo) * frame[y:y+h_logo, x:x+w_logo, c]
            )
    return frame

if run_detection and uploaded_file:
    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video path (default)
    output_path = "output_streamlit.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # UI elements
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Reset class counts
    class_counts = {name: 0 for name in model.names}

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model.predict(frame, verbose=False)[0]

        # Draw detections and count classes
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if label in class_counts:
                class_counts[label] += 1  # Internally maintained if you ever need it

        # Add logo
        if logo is not None:
            frame = overlay_logo(frame, logo)

        out.write(frame)

        # Show frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_placeholder.image(frame_image, caption=f"Frame {count + 1}/{total_frames}", use_container_width=True)

        count += 1
        progress_bar.progress(min(count / total_frames, 1.0))

    cap.release()
    out.release()

    st.success("✅ Detection complete!")
    st.video(output_path)
