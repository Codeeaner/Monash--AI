from ultralytics import YOLO
import os

def retrain_model():
    """Retrain the model with current ultralytics version using YOLOv8"""
    
    # Check if data exists
    data_path = "data.yaml"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return False
    
    print("Starting model retraining with YOLOv8n...")
    print("This will create a new model compatible with the current ultralytics version.")
    
    try:
        # Load a YOLOv8 nano model (compatible with current version)
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=data_path,
            epochs=50,  # Reduced epochs for faster training
            imgsz=416,
            batch=2,
            device='0' if os.system('nvidia-smi') == 0 else 'cpu',  # Use GPU if available
            workers=0,
            patience=10,
            project='runs/detect',
            name='retrain_v8',
            exist_ok=True,
            verbose=True
        )
        
        print("Training completed!")
        print(f"New model saved at: runs/detect/retrain_v8/weights/best.pt")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    retrain_model()