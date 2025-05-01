from ultralytics import YOLO, SAM
import cv2
import numpy as np
import torch

def segment_image(image_path):
    # Load YOLOv8 model
    yolo_model = YOLO('yolov8n.pt')
    
    # Load SAM model
    sam_model = SAM('sam_b.pt')
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Run YOLOv8 detection
    results = yolo_model(image)
    
    # Get detection boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Process each detected object with SAM
    segmented_results = []
    for box in boxes:
        # Convert box to integer coordinates
        box = box.astype(int)
        
        # Get SAM prediction for the box
        sam_result = sam_model(image, bboxes=[box])[0]
        segmented_results.append(sam_result)
    
    # Visualize results
    output_image = image.copy()
    
    # Draw segmentation masks with different colors
    for i, sam_result in enumerate(segmented_results):
        mask = sam_result.masks.data[0].cpu().numpy()
        
        # Create random color for this instance
        color = np.random.randint(0, 255, size=3).tolist()
        
        # Apply colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend with original image
        output_image = cv2.addWeighted(output_image, 1.0, colored_mask, 0.5, 0)
    
    # Save the result
    output_path = 'segmented_output.jpg'
    cv2.imwrite(output_path, output_image)
    print(f"Segmented image saved as {output_path}")

if __name__ == "__main__":
    # Use the provided image path
    image_path = "/home/rizwan/Work/Python/VS_agent/OIP.jpeg"
    try:
        segment_image(image_path)
    except Exception as e:
        print(f"Error: {str(e)}")