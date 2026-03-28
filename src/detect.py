from ultralytics import YOLO
import cv2
import os

# Path to trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs', 'detect', 'runs', 'train', 'pothole_detector', 'weights', 'best.pt')

def detect_image(image_path):
    model = YOLO(MODEL_PATH)
    results = model(image_path)
    results[0].save(filename='output.jpg')
    print("Detection complete! Output saved as output.jpg")

def detect_video(video_path):
    model = YOLO(MODEL_PATH)
    results = model(video_path, save=True)
    print("Video detection complete!")

if __name__ == '__main__':
    # Test on sample video
    video_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_video.mp4')
    detect_video(video_path)
