from ultralytics import YOLO
import os

from ultralytics import YOLO
import os

DATA_YAML = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.yaml')

def train():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=DATA_YAML,
        epochs=20,        
        imgsz=416,        
        batch=8,          
        name='pothole_detector',
        project='runs/train',
        exist_ok=True
    )

    print("Training complete!")
    print(f"Results saved to runs/train/pothole_detector")

if __name__ == '__main__':
    train()