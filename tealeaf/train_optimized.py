from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
import os
import gc
import psutil
import yaml
import platform

def optimize_system():
    """Optimize system settings for training"""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set GPU memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)  # Reduced from 0.9
    
    # Force garbage collection
    gc.collect()
    
    # Set environment variables for optimal performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
    os.environ['PYTHONHASHSEED'] = '0'  # For reproducibility
    
    # Windows-specific multiprocessing fix
    if platform.system() == 'Windows':
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
    
    # Set number of threads for CPU operations
    torch.set_num_threads(min(4, psutil.cpu_count() // 2))  # Reduced thread count

def get_optimal_batch_size(model, device, imgsz=640):
    """Dynamically determine optimal batch size - more conservative for Windows"""
    if device == 'cpu':
        return 1
    
    # Get GPU memory in GB
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # More conservative batch sizes for Windows multiprocessing
    if gpu_memory >= 24:  # RTX 4090, A6000, etc.
        return 16 if imgsz <= 640 else 8
    elif gpu_memory >= 16:  # RTX 4080, A5000, etc.
        return 12 if imgsz <= 640 else 6
    elif gpu_memory >= 12:  # RTX 4070 Ti, RTX 3080, etc.
        return 8 if imgsz <= 640 else 4
    elif gpu_memory >= 8:  # RTX 4060 Ti, RTX 3070, etc.
        return 6 if imgsz <= 640 else 3  # Reduced for your RTX 4060
    else:  # Lower memory GPUs
        return 4 if imgsz <= 640 else 2

def validate_data_yaml(data_path):
    """Validate and optimize data.yaml configuration"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in data.yaml")
    
    print(f"Dataset: {data['nc']} classes - {data['names']}")
    return data

def get_optimal_workers(batch_size):
    """Calculate optimal number of workers - Windows-specific optimization"""
    if platform.system() == 'Windows':
        # Windows has issues with multiprocessing, use fewer workers
        return 0  # Single-threaded dataloader for Windows
    else:
        cpu_count = psutil.cpu_count()
        optimal_workers = min(cpu_count, max(1, batch_size // 2))
        return optimal_workers

def main():
    # System optimization
    optimize_system()
    
    # Device selection with better GPU utilization
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = '0'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Model loading with better error handling and model selection
    model_paths = [
        r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train6\weights\best.pt",
        "yolo11n.pt",
        "yolov8n.pt", 
        "yolov5n.pt"
    ]
    
    model = None
    for model_path in model_paths:
        try:
            model = YOLO(model_path)
            print(f"Successfully loaded model: {model_path}")
            break
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
    
    if model is None:
        raise RuntimeError("Could not load any model")
    
    # Data validation
    data_path = r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\tealeaf\data.yaml"
    data_config = validate_data_yaml(data_path)
    
    # Dynamic parameter optimization (more conservative for Windows)
    imgsz = 640
    batch_size = get_optimal_batch_size(model, device, imgsz)
    workers = get_optimal_workers(batch_size)
    
    print(f"Optimized settings: batch_size={batch_size}, workers={workers}, imgsz={imgsz}")
    
    # Training with Windows-optimized parameters
    results = model.train(
        data=data_path,
        epochs=200,
        imgsz=640,
        batch=batch_size,
        workers=workers,  # Set to 0 for Windows
        device=device,
        
        # Training optimization
        patience=0,
        save=True,
        plots=True,
        cache=False,         # Disable RAM cache to reduce memory usage
        
        # Model optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation settings
        degrees=25,
        translate=0.1,
        scale=0.5,
        shear=0.1,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        
        # HSV augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Advanced training settings
        amp=True,
        close_mosaic=10,
        deterministic=False,
        single_cls=False,
        
        # Validation settings
        val=True,
        save_period=25,
        
        # Resume training capability
        resume=False,
        
        # Additional optimization flags
        profile=False,
        half=False,
        dfl=1.5,
        
        # Class balancing
        cls=0.5,
        box=7.5,
        
        # Multi-scale training
        rect=False,
        
        # Logging
        verbose=True,
        exist_ok=True,
        
        # Seed for reproducibility
        seed=42
    )
    
    # Post-training optimization
    print("\nTraining completed!")
    print(f"Best mAP@0.5: {results.box.map50:.4f}")
    print(f"Best mAP@0.5:0.95: {results.box.map:.4f}")
    
    # Clear cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

if __name__ == '__main__':
    freeze_support()  # Essential for Windows multiprocessing
    try:
        results = main()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()