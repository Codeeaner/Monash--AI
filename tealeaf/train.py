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
        model = YOLO(r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train3\weights\best.pt")
        print("Loaded yolo11n.pt (nano version)")
    except:
        try:
            model = YOLO("yolov8n.pt")
            print("Loaded yolov8n.pt (nano version)")
        except:
            model = YOLO("yolov5n.pt")
            print("Loaded yolov5n.pt (nano version)")

    # Train for 2000 epochs, start a new training from the weights (do not resume)
    results = model.train(
        data=r"C:\\Users\\amber\\OneDrive\\Documents\\GitHub\\Monash--AI\\tealeaf\\data.yaml",
        epochs=2000,         # 2000 epochs
        imgsz=640,
        batch=4,
        workers=0,
        device=device,
        patience=0,
        save=True,
        plots=True,
        cache=False,
        model=r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train3\weights\best.pt",
        degrees=25,          # Rotation augmentation
        flipud=0.5,          # Vertical flip augmentation
        fliplr=0.5,          # Horizontal flip augmentation
        hsv_h=0.015,         # HSV hue augmentation
        hsv_s=0.7,           # HSV saturation augmentation
        hsv_v=0.4,           # HSV value augmentation
        scale=0.5,           # Scale augmentation
        shear=0.1,           # Shear augmentation
        perspective=0.0,     # Perspective augmentation
        mosaic=1.0,          # Mosaic augmentation
        mixup=0.1            # Mixup augmentation
    )

    return results

if __name__ == '__main__':
    freeze_support()
    main()