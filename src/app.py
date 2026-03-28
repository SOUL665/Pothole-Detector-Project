import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs', 'detect', 'runs', 'train', 'pothole_detector', 'weights', 'best.pt')

model = YOLO(MODEL_PATH)

st.title("Pothole Detection System")
st.write("Upload a road image to detect potholes")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)
        result_image = results[0].plot()

    st.image(result_image, caption="Detected Potholes", use_column_width=True)
    
    num_potholes = len(results[0].boxes)
    st.success(f"Detected {num_potholes} pothole(s) in the image!")
