**Face Swapping App by Adil Khan**
This repository contains a Face Swapping application built with Streamlit and OpenCV, utilizing InsightFace for facial recognition and swapping. This app allows you to perform face swaps on both images and videos with minimal hassle.

**Features:**
Image Face Swap: Upload a source face image and a target image, and seamlessly swap faces between the two.
Video Face Swap: Upload a source face image and a target video, and apply the face swap across all frames in the video.
InsightFace Integration: Uses InsightFace's deep learning models to detect and swap faces with high precision.
Streamlit Interface: User-friendly web interface for easy interaction.
**How It Works:**
Image Swapping: Detects faces in both source and target images, then replaces the target face with the source face using inswapper_128.onnx.
Video Swapping: Swaps the face in each frame of the video by detecting faces in the target video and applying the face swap to match the source face.
**Installation:**
1. Clone the repository:
git clone https://github.com/adikhanofficial/face_swapping.git

2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run app.py OR Just Run run.py

**Requirements:**
Python 3.x
Streamlit
OpenCV
InsightFace

**Usage:**
Open the app, upload your source and target images or videos, and let the magic happen! Swapped images and videos can be downloaded directly from the app once processing is complete.

**Results**
<img src='https://raw.githubusercontent.com/AdiKhanOfficial/face_swapping/refs/heads/main/Results/Result.jpg' style='width:100%'/>
