from ultralytics import YOLO
import cv2


def run_inference(source="test.jpg", model_path=r"C:\Users\amber\OneDrive\Documents\GitHub\Monash--AI\runs\detect\train6\weights\best.pt", save=True, show=True):
    # Load the trained model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=source,        # Image, folder, video, webcam, etc.
        save=save,            # Save the results (image with boxes) to disk
        show=show,            # Show the results in a window
        conf=0.25,            # Confidence threshold
        imgsz=640             # Inference image size
    )

    # Count healthy vs unhealthy leaves
    healthy_count = 0
    unhealthy_count = 0
    
    # Process results to count leaf health status
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == 0:  # Unhealthy leaf
                    unhealthy_count += 1
                elif class_id == 1:  # Healthy leaf
                    healthy_count += 1
    
    # Display results
    print("\n" + "="*50)
    print("TEA LEAF HEALTH ANALYSIS RESULTS")
    print("="*50)
    print(f"🟢 Healthy leaves detected: {healthy_count}")
    print(f"🔴 Unhealthy leaves detected: {unhealthy_count}")
    print(f"📊 Total leaves detected: {healthy_count + unhealthy_count}")
    
    if healthy_count + unhealthy_count > 0:
        healthy_percentage = (healthy_count / (healthy_count + unhealthy_count)) * 100
        print(f"💚 Health percentage: {healthy_percentage:.1f}%")
    
    print("="*50)

    return results

if __name__ == "__main__":
    # Example usage:
    # Test on one image
    run_inference(source=r"C:\Users\amber\2025\T4G\data\test\images\IMG_3454_JPG.rf.b052a655c967962e1370f1fcecc29104.jpg")
