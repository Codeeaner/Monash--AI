from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import os

def main():
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set environment variable for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Check GPU availability
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = '0'  # Use GPU 0
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Load a COCO-pretrained YOLO model (using smaller version to save memory)
    try:
        model = YOLO(r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train8\weights\last.pt")
        print("Loaded yolo11n.pt (nano version)")
    except:
        try:
            model = YOLO("yolov8n.pt")
            print("Loaded yolov8n.pt (nano version)")
        except:
            model = YOLO("yolov5n.pt")
            print("Loaded yolov5n.pt (nano version)")
    
    # Train for 200 epochs, start a new training from the weights (do not resume)
    results = model.train(
        data=r"C:\\Users\\amber\\OneDrive\\Documents\\GitHub\\Monash--AI\\tealeaf\\data.yaml",
        epochs=200000,         # 200 epochs
        imgsz=416,          # Reduced image size
        batch=2,            # Very small batch size
        workers=0,
        device=device,      # Use detected device
        patience=0,       # Increased patience for early stopping                  
        save=True,
        plots=False,        # Disable plots to save memory
        cache=False,        # Disable caching to save memory
        model=r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train8\weights\last.pt"  # Use as starting weights
        # Do NOT use resume=True
    )
    
    return results

if __name__ == '__main__':
    freeze_support()
    main()