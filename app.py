
import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🚧",
    layout="wide"
)

# Load Trained YOLOv8 Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Application Title
st.title("🚧 Pothole and Road Damage Detection")
st.markdown("""
### CNN-Based Object Detection Using YOLOv8

Upload a road image or video to detect potholes and road damage.
""")

st.divider()

# Sidebar
st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Select Detection Mode",
    ["Image Detection", "Video Detection"]
)

# IMAGE DETECTION
if option == "Image Detection":

    st.header("🖼️ Image-Based Detection")

    uploaded_file = st.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Original Road Image",
            use_container_width=True
        )

        if st.button("Detect Potholes"):

            with st.spinner("Analyzing Image..."):

                results = model.predict(
                    image,
                    conf=0.25,
                    verbose=False
                )

                output_image = results[0].plot()

                output_image = cv2.cvtColor(
                    output_image,
                    cv2.COLOR_BGR2RGB
                )

                st.subheader("Detection Results")

                st.image(
                    output_image,
                    caption="Detected Road Damage",
                    use_container_width=True
                )

                count = len(results[0].boxes)

                st.success(
                    f"Total Detected Objects: {count}"
                )

# VIDEO DETECTION
elif option == "Video Detection":

    st.header("🎥 Video-Based Detection")

    uploaded_video = st.file_uploader(
        "Upload Road Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Detect Potholes in Video"):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(uploaded_video.read())
                input_path = temp_file.name

            output_path = tempfile.mktemp(
                suffix=".mp4"
            )

            cap = cv2.VideoCapture(input_path)

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 20

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            total_detections = 0
            frame_count = 0

            progress_bar = st.progress(0)
            status = st.empty()

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                results = model.predict(
                    frame,
                    conf=0.25,
                    verbose=False
                )

                detected_frame = results[0].plot()

                total_detections += len(
                    results[0].boxes
                )

                out.write(detected_frame)

                frame_count += 1

                if total_frames > 0:

                    progress_bar.progress(
                        min(frame_count / total_frames, 1.0)
                    )

                status.write(
                    f"Processing Frame: {frame_count}/{total_frames}"
                )

            cap.release()
            out.release()

            st.success("Video Detection Completed!")

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Frames Processed",
                frame_count
            )

            col2.metric(
                "Total Detections",
                total_detections
            )

            st.subheader("Processed Video")

            st.video(output_path)

            with open(output_path, "rb") as video:

                st.download_button(
                    "Download Detection Video",
                    data=video,
                    file_name="pothole_detection_output.mp4",
                    mime="video/mp4"
                )

            os.remove(input_path)

            os.remove(output_path)

# Footer
st.divider()

st.markdown(
    "Developed using Python, Streamlit, OpenCV and YOLOv8"
)
