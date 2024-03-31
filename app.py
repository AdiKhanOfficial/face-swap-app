import os
from tempfile import NamedTemporaryFile
import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import requests

app = ''
swapper = ''
st.set_page_config(page_title="FaceSwap App by Adil Khan")

def download_file(directory):
    url = "https://cdn.adikhanofficial.com/python/insightface/models/inswapper_128.onnx"
    filename = url.split('/')[-1]
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"{filename} already exists in the directory.")

def swap_faces(target_image, target_face, source_face):
    try:
        return swapper.get(target_image, target_face, source_face, paste_back=True)
    except Exception as e:
        st.error(f"Error during swaping: {e}")


def image_faceswap_app():
    st.title("Face Swapper for Image")
    source_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
    target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
    if source_image and target_image:
        with st.spinner("Swapping... Please wait."):
            try:
                source_image = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), -1)
                target_image = cv2.imdecode(np.frombuffer(target_image.read(), np.uint8), -1)
                source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)                
                source_faces = app.get(source_image)
                source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
                if len(source_faces) == 0:
                    raise ValueError("No faces found in the source image.")
                source_face = source_faces[0]
                target_faces = app.get(target_image)
                target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
                if len(target_faces) == 0:
                    raise ValueError("No faces found in the target image.")
                target_face = target_faces[0]
                swapped_image = swap_faces(target_image, target_face, source_face)                
                message_placeholder = st.empty()
                message_placeholder.success("Swapped Successfully!")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(source_image, caption="Source Image", use_column_width=True)
                with col2:
                    st.image(target_image, caption="Target Image", use_column_width=True)
                with col3:
                    st.image(swapped_image, caption="Swapped Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error during image processing: {e}")


def process_video(source_img, video_path, output_video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        source_faces = app.get(source_img)
        source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
        if len(source_faces) == 0:
            raise ValueError("No faces found in the source image.")
        source_face = source_faces[0]
        progress_placeholder = st.empty()
        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            target_faces = app.get(frame)
            target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
            if len(target_faces) > 0:
                frame = swap_faces(frame, target_faces[0], source_face)
            out.write(frame)
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time if elapsed_time > 0 else 0
            remaining_time_seconds = max(0, (total_frames - frame_count) / frames_per_second) if frames_per_second > 0 else 0
            remaining_minutes, remaining_seconds = divmod(remaining_time_seconds, 60)
            elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)
            progress_placeholder.text(
                f"Processed Frames: {frame_count}/{total_frames} | Elapsed Time: {int(elapsed_minutes)}m {int(elapsed_seconds)}s | Remaining Time: {int(remaining_minutes)}m {int(remaining_seconds)}s")
            frame_count += 1
        cap.release()
        out.release()
    except Exception as e:
        st.error(f"Error during video processing: {e}")


def video_faceswap_app():
    st.title("Face Swapper for Video")
    source_image = st.file_uploader("Upload Source Face Image", type=["jpg", "jpeg", "png"])
    if source_image is not None:
        source_image = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), -1)
    target_video = st.file_uploader("Upload Target Video", type=["mp4"])
    if target_video is not None:
        temp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(target_video.read())
        output_video_path = os.path.splitext(temp_video.name)[0] + '_output.mp4'
        status_placeholder = st.empty()
        try:
            with st.spinner("Processing... This may take a while."):
                process_video(source_image, temp_video.name, output_video_path)
            status_placeholder.success("Processing complete!")
            st.subheader("Your video is ready:")
            st.video(output_video_path)
        except Exception as e:
            st.error(f"Error during video processing: {e}")


def main():
    app_selection = st.sidebar.radio("Select App", ("Image Face Swapping", "Video Face Swapping"))
    if app_selection == "Image Face Swapping":
        image_faceswap_app()
    elif app_selection == "Video Face Swapping":
        video_faceswap_app()


if __name__ == "__main__":
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    root_dir = "/"
    download_file(root_dir)
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', root=root_dir,download=True, download_zip=True)
    main()
