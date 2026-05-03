import os
import json
import mimetypes
import time
import logging
from pathlib import Path
from typing import Dict, Any

import requests
import cv2
import numpy as np
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_HUGGINGFACE_SPACE_API_URL = "https://acwz-tealeafdetection.hf.space/run/predict"
DEFAULT_HUGGINGFACE_SPACE_BASE_URL = "https://acwz-tealeafdetection.hf.space"


class TeaLeafDetectionService:
    """Service for tea leaf detection using YOLO model."""

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):
        """
        Initialize the tea leaf detection service.

        Args:
            model_path: Unused. Kept for backward compatibility.
            confidence_threshold: Minimum confidence for detections
        """
        self.remote_space_url = self._resolve_remote_space_url()
        self.model_path = None
        self.confidence_threshold = confidence_threshold

        self.class_names = {0: "unhealthy", 1: "healthy"}
        self.colors = {
            0: (0, 0, 255),
            1: (0, 255, 0),
        }

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

            if not self.remote_space_url:
                return {
                    "error": "HUGGINGFACE_SPACE_API_URL is not configured",
                    "healthy_count": 0,
                    "unhealthy_count": 0,
                    "total_count": 0,
                    "health_percentage": 0.0,
                    "boxes": [],
                    "processing_time": time.time() - start_time,
                    "annotated_image_path": None,
                }

            logger.info("Running remote detection via Hugging Face Space API: %s", self.remote_space_url)
            return self._detect_image_via_huggingface_space(
                image_path=image_path,
                image=image,
                save_annotated=save_annotated,
                output_dir=output_dir,
                start_time=start_time,
            )

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

    def _detect_image_via_huggingface_space(self, image_path: str, image: np.ndarray,
                                            save_annotated: bool, output_dir: str,
                                            start_time: float) -> Dict[str, Any]:
        """Send the image directly to the Hugging Face Space /run/predict endpoint."""
        predict_url = f"{self.remote_space_url}/run/predict"
        with open(image_path, "rb") as image_file:
            content_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            files = {
                "data": (Path(image_path).name, image_file, content_type),
            }
            response = requests.post(
                predict_url,
                data={"confidence": str(self.confidence_threshold)},
                files=files,
                timeout=120,
            )
        response.raise_for_status()

        payload = response.json()
        detections = self._extract_remote_detections(payload)
        boxes_data = self._normalize_remote_boxes(detections)

        healthy_count = 0
        unhealthy_count = 0
        for box in boxes_data:
            class_id = box.get("class_id")
            class_name = str(box.get("class_name", "")).lower()

            if class_id == 0 or class_name == "unhealthy":
                unhealthy_count += 1
            elif class_id == 1 or class_name == "healthy":
                healthy_count += 1

        total_count = healthy_count + unhealthy_count
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0.0

        annotated_image_path = None
        if save_annotated:
            annotated_image_path = self._create_annotated_image(image, boxes_data, image_path, output_dir)

        logger.info(
            "Remote detection summary - Healthy: %s, Unhealthy: %s, Total: %s",
            healthy_count,
            unhealthy_count,
            total_count,
        )

        return {
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "total_count": total_count,
            "health_percentage": health_percentage,
            "boxes": boxes_data,
            "annotated_image_path": annotated_image_path,
            "processing_time": time.time() - start_time,
            "source": "huggingface_space_api",
        }

    def _resolve_remote_space_url(self) -> str:
        """Return the Space base URL, even if the environment variable includes /run/predict."""
        configured_url = os.getenv("HUGGINGFACE_SPACE_URL", "").strip() or os.getenv(
            "HUGGINGFACE_SPACE_API_URL", ""
        ).strip()
        if not configured_url:
            configured_url = DEFAULT_HUGGINGFACE_SPACE_BASE_URL

        configured_url = configured_url.rstrip("/")
        for suffix in ("/run/predict", "/call/predict"):
            if configured_url.endswith(suffix):
                configured_url = configured_url[: -len(suffix)]
        return configured_url

    def _extract_remote_detections(self, payload: Any) -> Any:
        """Extract the detections payload from a Gradio response."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return payload

        if isinstance(payload, list):
            return payload[-1] if payload else []

        if isinstance(payload, dict):
            for key in ("data", "result", "output", "outputs", "predictions", "detections"):
                value = payload.get(key)
                if value is None:
                    continue
                if isinstance(value, list):
                    return value[-1] if value else []
                if isinstance(value, dict):
                    nested = self._extract_remote_detections(value)
                    if nested is not value:
                        return nested
                    return value
                if isinstance(value, str):
                    try:
                        return self._extract_remote_detections(value)
                    except Exception:
                        return value

            return payload

        return payload

    def _normalize_remote_boxes(self, detections: Any) -> list:
        """Convert remote detections into the local box format used by the UI."""
        if isinstance(detections, str):
            try:
                detections = json.loads(detections)
            except json.JSONDecodeError:
                return []

        if isinstance(detections, dict):
            for key in ("detections", "predictions", "data", "items"):
                value = detections.get(key)
                if isinstance(value, list):
                    detections = value
                    break
            else:
                detections = [detections]

        if not isinstance(detections, list):
            return []

        boxes_data = []
        for detection in detections:
            if isinstance(detection, str):
                try:
                    detection = json.loads(detection)
                except json.JSONDecodeError:
                    continue

            if not isinstance(detection, dict):
                continue

            box = detection.get("box") if isinstance(detection.get("box"), dict) else {}
            class_id = detection.get("class")
            if class_id is None:
                class_id = detection.get("class_id")
            if isinstance(class_id, str) and class_id.isdigit():
                class_id = int(class_id)

            class_name = detection.get("name") or detection.get("class_name") or detection.get("label")
            confidence = detection.get("confidence")
            if confidence is None:
                confidence = detection.get("score")

            x1 = detection.get("x1", box.get("x1", box.get("xmin", box.get("left"))))
            y1 = detection.get("y1", box.get("y1", box.get("ymin", box.get("top"))))
            x2 = detection.get("x2", box.get("x2", box.get("xmax")))
            y2 = detection.get("y2", box.get("y2", box.get("ymax")))

            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue

            boxes_data.append({
                "class_id": int(class_id) if class_id is not None else None,
                "class_name": class_name or (self.class_names.get(int(class_id)) if class_id is not None else "unknown"),
                "confidence": float(confidence) if confidence is not None else 0.0,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            })

        return boxes_data

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
            "model_path": None,
            "model_loaded": False,
            "confidence_threshold": self.confidence_threshold,
            "class_names": self.class_names,
            "model_type": "Hugging Face Space API",
            "remote_api_url": self.remote_space_url,
        }
