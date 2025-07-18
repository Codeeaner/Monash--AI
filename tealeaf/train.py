from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import os
import psutil # To get CPU count for workers

def main():
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

    # Set environment variable for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("PYTORCH_CUDA_ALLOC_CONF set to 'expandable_segments:True'")

    # Check GPU availability
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = '0'  # Use GPU 0
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Load a COCO-pretrained YOLO model or your best.pt
    # The model loaded here will be used for training.
    # If you want to start fresh with a pretrained model (e.g., yolov8n.pt)
    # and transfer weights, you would load that directly here.
    try:
        model = YOLO(r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train3\weights\best.pt")
        print("Loaded model from C:\\Users\\amber\\OneDrive\\Documents\\GitHub\\Monash--AI\\runs\\detect\\train3\\weights\\best.pt")
    except Exception as e:
        print(f"Could not load specified model from train2: {e}. Trying default YOLOv8n/YOLOv5n.")
        try:
            model = YOLO("yolov8n.pt")
            print("Loaded yolov8n.pt (nano version)")
        except Exception as e:
            print(f"Could not load yolov8n.pt: {e}. Trying yolov5n.pt.")
            try:
                model = YOLO("yolov5n.pt")
                print("Loaded yolov5n.pt (nano version)")
            except Exception as e:
                print(f"Could not load yolov5n.pt: {e}. Exiting.")
                return # Exit if no model can be loaded

    # Determine optimal workers based on CPU count
    # Using half of the available CPU cores is a good starting point
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
    if num_workers == 0:
        print("Warning: Could not determine CPU count, defaulting workers to 0. This may slow down data loading.")
    else:
        print(f"Using {num_workers} workers for data loading.")

    # Train the model
    results = model.train(
        data=r"C:\\Users\\amber\\OneDrive\\Documents\\GitHub\\Monash--AI\\tealeaf\\data.yaml",
        epochs=2000,
        imgsz=640,
        batch=16,  # Increased batch size to 16 for more stable gradients
        workers=num_workers, # Set workers to use multiple CPU cores for faster data loading
        device=device,
        patience=0,
        save=True,
        plots=True,
        cache=False,
        degrees=25,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        shear=0.1,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.1,
        lr0=0.01,  # Initial learning rate (default for YOLOv8 is often 0.01)
        lrf=0.01,   # Final learning rate (relative to lr0, default is often 0.01 for 1% of lr0)
        optimizer="SGD"
    )

    return results

if __name__ == '__main__':
    freeze_support()
    main()
