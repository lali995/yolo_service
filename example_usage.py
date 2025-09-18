#!/usr/bin/env python3
"""
Example demonstrating the complete workflow:
1. Load an image
2. Convert to RGBA
3. Send to YOLO service
4. Parse JSON results
5. Render bounding boxes
6. Save result
"""

import json
import numpy as np
import cv2
from yolo_service import init_engine, detect_rgba_image, shutdown

def process_image_with_yolo(input_path, output_path, engine_path="./yolo11n.engine"):
    """
    Complete image processing workflow using RGBA service
    """
    print(f"Processing image: {input_path}")
    
    # Step 1: Load image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image {input_path}")
        return False
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} channels: {image.shape[2]}")
    
    # Step 2: Convert to RGBA
    rgba_image = cv2.cvtColor(image, cv::COLOR_BGR2RGBA)
    print(f"Converted to RGBA: {rgba_image.shape}")
    
    # Step 3: Initialize YOLO service
    print("Initializing YOLO service...")
    if not init_engine(engine_path, imgsz=640, conf=0.25, device=0):
        print("Error: Failed to initialize YOLO model")
        return False
    
    # Step 4: Send RGBA data to YOLO service
    print("Running YOLO detection...")
    rgba_flat = rgba_image.flatten()
    result_json = detect_rgba_image(rgba_flat, rgba_image.shape[1], rgba_image.shape[0])
    
    # Step 5: Parse JSON results
    result = json.loads(result_json)
    print(f"Detection result: {json.dumps(result, indent=2)}")
    
    if not result["ok"]:
        print(f"Detection failed: {result['error']}")
        return False
    
    # Step 6: Render bounding boxes
    print(f"Found {result['count']} detections")
    result_image = image.copy()
    
    for i, detection in enumerate(result["detections"]):
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        
        print(f"  {i+1}. {class_name} (confidence: {confidence:.3f})")
        print(f"     BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) to ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
        
        # Draw bounding box
        pt1 = (int(bbox["x1"]), int(bbox["y1"]))
        pt2 = (int(bbox["x2"]), int(bbox["y2"]))
        cv2.rectangle(result_image, pt1, pt2, (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_pt = (int(bbox["x1"]), int(bbox["y1"]) - 10)
        if label_pt[1] < 0:
            label_pt = (int(bbox["x1"]), int(bbox["y1"]) + 20)
        
        # Draw label background
        cv2.rectangle(result_image,
                     (label_pt[0], label_pt[1] - label_size[1] - 5),
                     (label_pt[0] + label_size[0], label_pt[1] + 5),
                     (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(result_image, label, label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Step 7: Save result
    success = cv2.imwrite(output_path, result_image)
    if success:
        print(f"Result saved to: {output_path}")
    else:
        print(f"Error: Failed to save result to {output_path}")
        return False
    
    # Save JSON for debugging
    json_path = output_path + ".json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"JSON result saved to: {json_path}")
    
    return True

def main():
    print("=== YOLO Image Processing Example ===")
    
    # Example usage
    input_image = "test_image.jpg"  # Change this to your image
    output_image = "result_with_boxes.jpg"
    engine_path = "./yolo11n.engine"  # Change this to your engine path
    
    success = process_image_with_yolo(input_image, output_image, engine_path)
    
    if success:
        print("Image processing completed successfully!")
    else:
        print("Image processing failed!")
    
    # Cleanup
    shutdown()

if __name__ == "__main__":
    main()
