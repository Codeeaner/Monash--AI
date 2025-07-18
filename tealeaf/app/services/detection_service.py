import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeaLeafDetectionService:
    """Service for tea leaf detection using YOLO model."""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):
        """
        Initialize the tea leaf detection service.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        if model_path is None:
            # Use the existing model path from the project
            model_path = r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train3\weights\best.pt"
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False
        
        # Class mapping based on data.yaml
        self.class_names = {0: "unhealthy", 1: "healthy"}
        
        # Colors for visualization (BGR format)
        self.colors = {
            0: (0, 0, 255),    # Red for unhealthy
            1: (0, 255, 0),    # Green for healthy
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading YOLO model from: {self.model_path}")
            
            # Try to load the model with proper error handling
            import torch
            
            # Set environment variable to allow loading custom models
            os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
            
            # Try direct loading first
            try:
                self.model = YOLO(self.model_path)
                self.model_loaded = True
                logger.info("✅ Model loaded successfully")
                
                # Test the model by running a dummy prediction to ensure it works
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = self.model.predict(dummy_image, verbose=False)
                logger.info("✅ Model test prediction successful")
                return
                
            except Exception as e1:
                logger.error(f"❌ Direct model loading failed: {e1}")
                
                # Try alternative loading method
                try:
                    # Load with explicit weights_only=False
                    checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                    
                    # Create new YOLO model and load the weights
                    self.model = YOLO('yolov8n.pt')  # Start with base model
                    
                    # Load the state dict from checkpoint
                    if 'model' in checkpoint:
                        self.model.model.load_state_dict(checkpoint['model'].state_dict())
                    else:
                        self.model.model.load_state_dict(checkpoint)
                    
                    self.model_loaded = True
                    logger.info("✅ Model loaded successfully with manual weight loading")
                    
                    # Test the model
                    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                    test_results = self.model.predict(dummy_image, verbose=False)
                    logger.info("✅ Model test prediction successful")
                    return
                    
                except Exception as e2:
                    logger.error(f"❌ Alternative model loading failed: {e2}")
                    
                    # Final attempt: try loading as different YOLO versions
                    try:
                        # Try loading as YOLOv8 explicitly
                        from ultralytics import YOLOv8
                        self.model = YOLOv8(self.model_path)
                        self.model_loaded = True
                        logger.info("✅ Model loaded as YOLOv8")
                        return
                        
                    except Exception as e3:
                        logger.error(f"❌ YOLOv8 loading failed: {e3}")
                        
                        # Last resort: try with different model formats
                        try:
                            # Check if it's an ONNX model
                            if self.model_path.endswith('.onnx'):
                                self.model = YOLO(self.model_path)
                                self.model_loaded = True
                                logger.info("✅ Model loaded as ONNX")
                                return
                            
                            # Check if it's a TensorRT model
                            if self.model_path.endswith('.engine'):
                                self.model = YOLO(self.model_path)
                                self.model_loaded = True
                                logger.info("✅ Model loaded as TensorRT")
                                return
                                
                        except Exception as e4:
                            logger.error(f"❌ All model loading attempts failed: {e4}")
            
            # If all loading attempts failed
            self.model = None
            self.model_loaded = False
            logger.error("❌ Failed to load model - all attempts exhausted")
            
        except Exception as e:
            logger.error(f"❌ Critical error during model loading: {e}")
            self.model = None
            self.model_loaded = False
    
    def detect_image(self, image_path: str, save_annotated: bool = True,
                    output_dir: str = "results") -> Dict[str, Any]:
        """
        Detect tea leaves in a single image.
        
        Args:
            image_path: Path to the input image
            save_annotated: Whether to save annotated image
            output_dir: Directory to save results
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        try:
            # Validate image file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Read and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Check if model is loaded
            if not self.model_loaded or self.model is None:
                logger.error("❌ Model not loaded - cannot perform detection")
                return {
                    "error": "Model not loaded",
                    "healthy_count": 0,
                    "unhealthy_count": 0,
                    "total_count": 0,
                    "health_percentage": 0.0,
                    "boxes": [],
                    "processing_time": time.time() - start_time,
                    "annotated_image_path": None
                }
            
            logger.info(f"🔍 Running YOLO detection on: {image_path}")
            
            # Run inference using the YOLO model
            results = self.model.predict(
                source=image_path,
                conf=self.confidence_threshold,
                save=False,  # We'll handle saving ourselves
                show=False,
                verbose=False,
                imgsz=640
            )
            
            logger.info(f"✅ YOLO detection completed for: {image_path}")
            
            # Process results
            detection_data = self._process_yolo_results(results, image, save_annotated, output_dir, image_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            detection_data["processing_time"] = processing_time
            
            logger.info(f"📊 Detection summary - Healthy: {detection_data['healthy_count']}, "
                       f"Unhealthy: {detection_data['unhealthy_count']}, "
                       f"Total: {detection_data['total_count']}")
            
            return detection_data
            
        except Exception as e:
            logger.error(f"❌ Error processing image {image_path}: {e}")
            return {
                "error": str(e),
                "healthy_count": 0,
                "unhealthy_count": 0,
                "total_count": 0,
                "health_percentage": 0.0,
                "boxes": [],
                "processing_time": time.time() - start_time,
                "annotated_image_path": None
            }

    def _process_yolo_results(self, results, image: np.ndarray, save_annotated: bool, 
                             output_dir: str, image_path: str) -> Dict[str, Any]:
        """Process YOLO detection results."""
        
        healthy_count = 0
        unhealthy_count = 0
        boxes_data = []
        
        logger.info(f"🔄 Processing {len(results)} result(s)")
        
        # Process each result
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                logger.info(f"📦 Found {len(result.boxes)} detection(s)")
                
                for box in result.boxes:
                    # Extract class ID and confidence
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    
                    # Skip low confidence detections
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Count by class
                    if class_id == 0:  # Unhealthy leaf
                        unhealthy_count += 1
                    elif class_id == 1:  # Healthy leaf
                        healthy_count += 1
                    
                    # Store box data
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    boxes_data.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    })
                    
                    logger.info(f"  📋 {class_name}: {confidence:.3f} at ({x1},{y1},{x2},{y2})")
            else:
                logger.info("📭 No detections found in this image")
        
        # Calculate statistics
        total_count = healthy_count + unhealthy_count
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0.0
        
        # Create annotated image if requested (always save, even with no detections)
        annotated_image_path = None
        if save_annotated:
            annotated_image_path = self._create_annotated_image(image, boxes_data, image_path, output_dir)
        
        return {
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "total_count": total_count,
            "health_percentage": health_percentage,
            "boxes": boxes_data,
            "annotated_image_path": annotated_image_path
        }

    def _create_annotated_image(self, image: np.ndarray, boxes_data: List[Dict], 
                               original_path: str, output_dir: str) -> str:
        """Create annotated image with bounding boxes."""
        
        try:
            # Create a copy of the image for annotation
            annotated_image = image.copy()
            
            # Draw each bounding box (only if we have detections)
            if boxes_data:
                for box in boxes_data:
                    class_id = box["class_id"]
                    class_name = box["class_name"]
                    confidence = box["confidence"]
                    x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
                    
                    # Get color for this class
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Calculate label size and position
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Ensure label fits within image bounds
                    label_y = max(y1, label_height + 10)
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_image,
                        (x1, label_y - label_height - 10),
                        (x1 + label_width, label_y),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
            else:
                # Add "No detections" text if no boxes found
                text = "No tea leaves detected"
                font_scale = 1.0
                thickness = 2
                color = (0, 0, 255)  # Red color
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Calculate position (center of image)
                img_height, img_width = annotated_image.shape[:2]
                x = (img_width - text_width) // 2
                y = (img_height + text_height) // 2
                
                # Draw text background
                cv2.rectangle(
                    annotated_image,
                    (x - 10, y - text_height - 10),
                    (x + text_width + 10, y + 10),
                    (255, 255, 255),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated_image,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness
                )
            
            # Save the annotated image
            return self._save_annotated_image(annotated_image, original_path, output_dir)
            
        except Exception as e:
            logger.error(f"❌ Failed to create annotated image: {e}")
            return None

    def _save_annotated_image(self, image: np.ndarray, original_path: str, 
                             output_dir: str) -> str:
        """Save annotated image to output directory."""
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename
            original_name = Path(original_path).stem
            annotated_filename = f"{original_name}_annotated.jpg"
            annotated_path = os.path.join(output_dir, annotated_filename)
            
            # Save image
            success = cv2.imwrite(annotated_path, image)
            
            if success:
                logger.info(f"💾 Annotated image saved: {annotated_path}")
                return annotated_path
            else:
                logger.error(f"❌ Failed to save annotated image to: {annotated_path}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error saving annotated image: {e}")
            return None

    def print_results(self, results: Dict[str, Any]):
        """Print detection results in a formatted way."""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n" + "="*50)
        print("TEA LEAF HEALTH ANALYSIS RESULTS")
        print("="*50)
        print(f"🟢 Healthy leaves detected: {results['healthy_count']}")
        print(f"🔴 Unhealthy leaves detected: {results['unhealthy_count']}")
        print(f"📊 Total leaves detected: {results['total_count']}")
        
        if results['total_count'] > 0:
            print(f"💚 Health percentage: {results['health_percentage']:.1f}%")
            print(f"⏱️ Processing time: {results['processing_time']:.2f}s")
            
            if results['annotated_image_path']:
                print(f"🖼️ Annotated image: {results['annotated_image_path']}")
        else:
            print("ℹ️ No tea leaves detected in the image")
        
        print("="*50)
    
    def detect_batch(self, image_paths: List[str], output_dir: str = "results",
                    progress_callback=None) -> List[Dict[str, Any]]:
        """
        Detect tea leaves in multiple images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of detection results
        """
        results = []
        total_images = len(image_paths)
        
        logger.info(f"🚀 Starting batch detection on {total_images} images")
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"📸 Processing image {i+1}/{total_images}: {image_path}")
            
            # Detect on single image
            result = self.detect_image(image_path, save_annotated=True, output_dir=output_dir)
            result["image_path"] = image_path
            result["image_name"] = Path(image_path).name
            results.append(result)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, total_images, Path(image_path).name)
        
        logger.info(f"✅ Batch detection completed on {total_images} images")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        
        return {
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "confidence_threshold": self.confidence_threshold,
            "class_names": self.class_names,
            "model_type": str(type(self.model)) if self.model else "None"
        }
    
    def is_model_loaded(self) -> bool:
        """Check if the model is properly loaded."""
        return self.model_loaded and self.model is not None
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for detections."""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold set to: {threshold}")
        else:
            logger.warning(f"Invalid confidence threshold: {threshold}. Must be between 0.0 and 1.0")
    
    def _trigger_analytics(self, detection_data: Dict[str, Any], image_path: str):
        """
        Trigger analytics for detection result.
        
        Args:
            detection_data: Detection results
            image_path: Path to the original image
        """
        try:
            from app.services.analytics_service import AnalyticsService
            
            # Check if annotated image exists
            annotated_image_path = detection_data.get("annotated_image_path")
            if not annotated_image_path or not os.path.exists(annotated_image_path):
                logger.warning(f"No annotated image found for analytics: {annotated_image_path}")
                return
            
            # Initialize analytics service
            analytics_service = AnalyticsService()
            
            # Run analysis (this will be done in background in production)
            analysis_result = analytics_service.analyze_detection_result(
                detection_data, annotated_image_path
            )
            
            logger.info(f"Analytics completed for {image_path}: {analysis_result.get('analysis_id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to trigger analytics for {image_path}: {e}")
    
    def enable_analytics_mode(self):
        """Enable automatic analytics after detection."""
        self.enable_analytics = True
        logger.info("Analytics mode enabled")
    
    def disable_analytics_mode(self):
        """Disable automatic analytics after detection."""
        self.enable_analytics = False
        logger.info("Analytics mode disabled")
