from ultralytics import YOLO
import cv2
import os
from pathlib import Path


def run_inference(source="test.jpg", 
                 original_model_path=r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train3\weights\best.pt",
                 classification_model_path=r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\tealeaf\runs\detect\train\weights\best.pt", 
                 save=True, show=True, crop_boxes=True):
    
    # Load both models
    print("Loading models...")
    original_model = YOLO(original_model_path)  # For cropping tea leaves
    classification_model = YOLO(classification_model_path)  # For health classification
    print("Models loaded successfully!")

    # Run inference with original model to detect and crop tea leaves
    print("\nStep 1: Detecting tea leaves for cropping...")
    results = original_model.predict(
        source=source,        # Image, folder, video, webcam, etc.
        save=save,            # Save the results (image with boxes) to disk
        show=show,            # Show the results in a window
        conf=0.25,            # Confidence threshold
        imgsz=640             # Inference image size
    )

    # Count total leaves detected by original model
    total_leaves_detected = 0
    
    # Create output directories for cropped images
    source_path = Path(source)
    output_dir = source_path.parent / f"{source_path.stem}_crops"
    annotated_dir = output_dir / "annotated"
    
    if crop_boxes:
        output_dir.mkdir(exist_ok=True)
        annotated_dir.mkdir(exist_ok=True)
    
    # Load the original image for cropping
    original_image = cv2.imread(source)
    if original_image is None:
        print(f"Error: Could not load image from {source}")
        return results
    
    cropped_images = []  # Store paths of cropped images for health classification
    
    # Process results to crop detected tea leaves
    for result_idx, result in enumerate(results):
        if result.boxes is not None:
            for box_idx, box in enumerate(result.boxes):
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                total_leaves_detected += 1
                
                # Crop the detected box
                if crop_boxes:
                    # Add some padding to the crop (optional)
                    padding = 10
                    x1_crop = max(0, x1 - padding)
                    y1_crop = max(0, y1 - padding)
                    x2_crop = min(original_image.shape[1], x2 + padding)
                    y2_crop = min(original_image.shape[0], y2 + padding)
                    
                    # Crop the image
                    cropped_leaf = original_image[y1_crop:y2_crop, x1_crop:x2_crop]
                    
                    # Save the cropped image
                    crop_filename = f"leaf_crop_{box_idx}.jpg"
                    crop_path = output_dir / crop_filename
                    
                    success = cv2.imwrite(str(crop_path), cropped_leaf)
                    if success:
                        print(f"Saved cropped leaf: {crop_path}")
                        cropped_images.append(crop_path)
                    else:
                        print(f"Failed to save cropped image: {crop_path}")

    # Step 2: Run health classification on cropped images
    healthy_count = 0
    unhealthy_count = 0
    
    if crop_boxes and cropped_images:
        print(f"\nStep 2: Running health classification on {len(cropped_images)} cropped leaves...")
        print(f"{'='*50}")
        
        for crop_path in cropped_images:
            try:
                # Run health classification on the cropped image
                crop_results = classification_model.predict(
                    source=str(crop_path),
                    save=False,  # Don't save to default location
                    show=False,  # Don't show results
                    conf=0.25,
                    imgsz=640
                )
                
                # Load the cropped image
                crop_img = cv2.imread(str(crop_path))
                if crop_img is None:
                    print(f"Error: Could not load cropped image {crop_path}")
                    continue
                
                # Draw bounding boxes on the cropped image
                annotated_img = crop_img.copy()
                crop_healthy = 0
                crop_unhealthy = 0
                
                for crop_result in crop_results:
                    if crop_result.boxes is not None:
                        for box in crop_result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Count detections in cropped image
                            if class_id == 0:  # Unhealthy
                                crop_unhealthy += 1
                                unhealthy_count += 1
                                color = (0, 0, 255)  # Red for unhealthy
                                label = f"Unhealthy {confidence:.2f}"
                            elif class_id == 1:  # Healthy
                                crop_healthy += 1
                                healthy_count += 1
                                color = (0, 255, 0)  # Green for healthy
                                label = f"Healthy {confidence:.2f}"
                            else:
                                color = (255, 0, 0)  # Blue for unknown
                                label = f"Class_{class_id} {confidence:.2f}"
                            

                            # Draw bounding box
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                            

                            # Draw label
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), color, -1)
                            cv2.putText(annotated_img, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # If no health classification detected, assume the whole crop is one leaf
                if crop_healthy == 0 and crop_unhealthy == 0:
                    # You could add logic here to classify the entire crop as healthy/unhealthy
                    # For now, we'll mark it as unclassified
                    print(f"No health classification detected in {crop_path.name}")
                
                # Save annotated cropped image
                annotated_filename = f"annotated_{crop_path.name}"
                annotated_path = annotated_dir / annotated_filename
                
                success = cv2.imwrite(str(annotated_path), annotated_img)
                if success:
                    print(f"Saved annotated crop: {annotated_path} (H:{crop_healthy}, U:{crop_unhealthy})")
                else:
                    print(f"Failed to save annotated crop: {annotated_path}")
                    
            except Exception as e:
                print(f"Error processing cropped image {crop_path}: {e}")
    
    # Display results
    print("\n" + "="*50)
    print("TEA LEAF HEALTH ANALYSIS RESULTS")
    print("="*50)
    print(f"📋 Step 1 - Original Model (Leaf Detection):")
    print(f"   🍃 Total tea leaves detected: {total_leaves_detected}")
    print(f"📋 Step 2 - Classification Model (Health Analysis):")
    print(f"   🟢 Healthy leaves detected: {healthy_count}")
    print(f"   🔴 Unhealthy leaves detected: {unhealthy_count}")
    print(f"   📊 Total classified leaves: {healthy_count + unhealthy_count}")
    
    if healthy_count + unhealthy_count > 0:
        healthy_percentage = (healthy_count / (healthy_count + unhealthy_count)) * 100
        print(f"   💚 Health percentage: {healthy_percentage:.1f}%")
    
    if crop_boxes and total_leaves_detected > 0:
        print(f"📁 Cropped images saved to: {output_dir}")
        print(f"📸 Total cropped images: {len(cropped_images)}")
        print(f"🎨 Annotated crops saved to: {annotated_dir}")
    
    print("="*50)

    return results

if __name__ == "__main__":
    # Example usage:
    # Test on one image with both models
    run_inference(
        source=r"C:\Users\amber\Downloads\camellia_sinensis.webp",
        original_model_path=r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train3\weights\best.pt",  # For cropping
        classification_model_path=r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\tealeaf\runs\detect\train\weights\best.pt",  # For health classification
        crop_boxes=True  # Enable cropping and classification
    )
