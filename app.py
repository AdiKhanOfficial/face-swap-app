import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

app = ''
swapper = ''

def download_inswapper():
    url = 'https://cdn.adikhanofficial.com/python/insightface/models/inswapper_128.onnx'
    destination_path = '/home/appuser/.insightface/models/inswapper_128.onnx'
    if not os.path.exists(destination_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, 'wb') as file:
            file.write(requests.get(url).content)

#process image for face swapping
def swap_faces(source_image, target_image):
    source_faces = app.get(source_image)
    source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
    assert len(source_faces) > 0
    source_face = source_faces[0]
    target_faces = app.get(target_image)
    target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
    assert len(target_faces) > 0
    target_face = target_faces[0]

    new_image = swapper.get(target_image, target_face, source_face, paste_back=True)
    return new_image

#process video for face swapping
def process_video(source_img, video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Detect Source Face
    reference_faces = app.get(source_img)
    reference_faces = sorted(reference_faces, key=lambda x: x.bbox[0])
    assert len(reference_faces) > 0
    source_face = reference_faces[0]
    progress_placeholder = st.empty()
    frame_count = 0
    start_time = time.time()
    video_placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        faces = sorted(faces, key=lambda x: x.bbox[0])
        if len(faces) > 0:
            frame = swapper.get(frame, faces[0], source_face, paste_back=True)
        out.write(frame)
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Visualized Image', use_column_width=True)
        elapsed_time = time.time() - start_time
        frames_per_second = frame_count / elapsed_time if elapsed_time > 0 else 0
        remaining_time_seconds = max(0,
                                     (total_frames - frame_count) / frames_per_second) if frames_per_second > 0 else 0
        remaining_minutes, remaining_seconds = divmod(remaining_time_seconds, 60)
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)

        progress_placeholder.text(
            f"Processed Frames: {frame_count}/{total_frames} | Elapsed Time: {int(elapsed_minutes)}m {int(elapsed_seconds)}s | Remaining Time: {int(remaining_minutes)}m {int(remaining_seconds)}s")
        frame_count += 1

    cap.release()
    out.release()

#run image swapper interface
def image_faceswap_app():
    st.title("Face Swapper App")
    source_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
    target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

    if source_image and target_image:
        message_placeholder = st.empty()
        message_placeholder.info("Swaping...")
        col1, col2, col3 = st.columns([1, 1, 1])
        source_image_array = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), -1)
        target_image_array = cv2.imdecode(np.frombuffer(target_image.read(), np.uint8), -1)
        source_image_array_rgb = cv2.cvtColor(source_image_array, cv2.COLOR_BGR2RGB)
        target_image_array_rgb = cv2.cvtColor(target_image_array, cv2.COLOR_BGR2RGB)

        swapped_image = swap_faces(source_image_array_rgb, target_image_array_rgb)

        with col1:
            st.image(source_image_array_rgb, caption="Source Image", use_column_width=True)
        with col2:
            st.image(target_image_array_rgb, caption="Target Image", use_column_width=True)
        with col3:
            st.image(swapped_image, caption="Swapped Image", use_column_width=True)
        message_placeholder.success("Swapped Successfully!")

#run video swapper interface
def video_faceswap_app():
    st.title("Face Swapper for Video")
    source_image = st.file_uploader("Upload Source Face Image", type=["jpg", "jpeg", "png"])
    if source_image is not None:
        source_img = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), -1)

    target_video = st.file_uploader("Upload Target Video", type=["mp4"])
    if target_video is not None:
        temp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(target_video.read())
        output_video_path = os.path.splitext(temp_video.name)[0] + '_output.mp4'

        # Process Video Face Swapping
        status_placeholder = st.empty()
        with st.spinner("Processing... This may take a while."):
            process_video(source_img, temp_video.name, output_video_path)
        status_placeholder.success("Processing complete!")
        st.subheader("Result Video:")
        st.video(output_video_path)
    

def main():
    app_selection = st.sidebar.radio("Select App", ("Image Face Swapping App", "Video Face Swapping App"))
    if app_selection == "Image Face Swapping App":
        image_faceswap_app()
    elif app_selection == "Video Face Swapping App":
        video_faceswap_app()


if __name__ == "__main__":
    download_inswapper()  
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', root='/home/appuser/.insightface', download=True, download_zip=True)
    main()
