# YOLOv8-SAM-Segmentation

This project combines YOLOv8 and Segment Anything Model (SAM) for advanced image segmentation.

## Project Structure
- `segment_anything.py`: Main script for image segmentation
- Model files (not included in repository):
  - `sam_b.pt`: SAM model weights
  - `yolov8n.pt`: YOLOv8 model weights
- Sample images:
  - `OIP.jpeg`: Input image
  - `segmented_output.jpg`: Output segmented image

## Setup
1. Clone this repository
2. Download the required model weights:
   - SAM model weights (`sam_b.pt`)
   - YOLOv8 model weights (`yolov8n.pt`)
3. Run `segment_anything.py` with your input images

Note: Model weight files (*.pt) are not included in the repository due to size constraints.