# Pothole Detection System
A computer vision project that detects potholes in road images and videos using YOLOv8.

## Problem Statement
Potholes are a major cause of road accidents and vehicle damage. Manual inspection of roads is slow and costly. This project automates pothole detection using deep learning.

## Features
- Detects potholes in images with bounding boxes
- Works on video files frame by frame
- Simple web interface built with Streamlit

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit

## Setup Instructions

1. Clone the repository
   git clone https://github.com/SOUL665/Pothole-Detector-Project.git

2. Create a virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Run the app
   streamlit run src/app.py

## Project Structure
- data/ - Dataset with train and validation images
- src/train.py - Model training script
- src/detect.py - Detection script for images and videos
- src/app.py - Streamlit web application
- runs/ - Training results and model weights

## Results
- mAP50: 0.715
- Precision: 0.74
- Recall: 0.646
