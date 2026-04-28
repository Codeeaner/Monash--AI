import os
import time
import logging
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "runs" / "detect" / "train3" / "weights" / "best.pt"


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
            env_model_path = os.getenv("MODEL_PATH")
            if env_model_path:
                env_candidate = Path(env_model_path).expanduser()
                if env_candidate.exists():
                    model_path = str(env_candidate)
                    logger.info(f"Using MODEL_PATH from environment: {model_path}")
                else:
                    logger.warning(f"MODEL_PATH is set but missing: {env_candidate}")

        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)
            logger.info(f"Using default model path: {model_path}")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False

        self.class_names = {0: "unhealthy", 1: "healthy"}
        self.colors = {
            0: (0, 0, 255),
            1: (0, 255, 0),
        }

        self._load_model()

    def _load_model(self):
        """Load the YOLO model via Ultralytics only."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model_loaded = True
            logger.info("Model loaded successfully")

            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_image, verbose=False)
            logger.info("Model test prediction successful")

        except Exception as e:
            logger.error(f"Critical error during model loading: {e}")
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
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            if not self.model_loaded or self.model is None:
                logger.error("Model not loaded - cannot perform detection")
                return {
                    "error": "Model not loaded",
                    "healthy_count": 0,
                    "unhealthy_count": 0,
                    "total_count": 0,
                    "health_percentage": 0.0,
                    "boxes": [],
                    "processing_time": time.time() - start_time,
                    "annotated_image_path": None,
                }

            logger.info(f"Running YOLO detection on: {image_path}")

            results = self.model.predict(
                source=image_path,
                conf=self.confidence_threshold,
                save=False,
                show=False,
                verbose=False,
                imgsz=640,
            )

            detection_data = self._process_yolo_results(results, image, save_annotated, output_dir, image_path)
            detection_data["processing_time"] = time.time() - start_time

            logger.info(
                "Detection summary - Healthy: %s, Unhealthy: %s, Total: %s",
                detection_data["healthy_count"],
                detection_data["unhealthy_count"],
                detection_data["total_count"],
            )
            return detection_data

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "error": str(e),
                "healthy_count": 0,
                "unhealthy_count": 0,
                "total_count": 0,
                "health_percentage": 0.0,
                "boxes": [],
                "processing_time": time.time() - start_time,
                "annotated_image_path": None,
            }

    def _process_yolo_results(self, results, image: np.ndarray, save_annotated: bool,
                             output_dir: str, image_path: str) -> Dict[str, Any]:
        """Process YOLO detection results."""
        healthy_count = 0
        unhealthy_count = 0
        boxes_data = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    if confidence < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if class_id == 0:
                        unhealthy_count += 1
                    elif class_id == 1:
                        healthy_count += 1

                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    boxes_data.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    })

        total_count = healthy_count + unhealthy_count
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0.0

        annotated_image_path = None
        if save_annotated:
            annotated_image_path = self._create_annotated_image(image, boxes_data, image_path, output_dir)

        return {
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "total_count": total_count,
            "health_percentage": health_percentage,
            "boxes": boxes_data,
            "annotated_image_path": annotated_image_path,
        }

    def _create_annotated_image(self, image: np.ndarray, boxes_data: list,
                               original_image_path: str, output_dir: str) -> str:
        """Create and save annotated image with detection boxes."""
        os.makedirs(output_dir, exist_ok=True)

        annotated_image = image.copy()

        for box_data in boxes_data:
            x1, y1, x2, y2 = int(box_data["x1"]), int(box_data["y1"]), int(box_data["x2"]), int(box_data["y2"])
            class_id = box_data["class_id"]
            class_name = box_data["class_name"]
            confidence = box_data["confidence"]

            color = self.colors.get(class_id, (255, 255, 255))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        original_name = Path(original_image_path).stem
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"{original_name}_{unique_id}_annotated.jpg"
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, annotated_image)
        return output_path

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "confidence_threshold": self.confidence_threshold,
            "class_names": self.class_names,
            "model_type": "YOLO",
        }
