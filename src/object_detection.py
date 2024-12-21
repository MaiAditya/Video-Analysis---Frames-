# src/object_detection.py
from ultralytics import YOLO
import cv2
import os
from typing import List, Optional

def detect_objects(frames: List[str], model_path: str = "yolov8n.pt", 
                  output_dir: str = "results/detections") -> List[Optional[dict]]:
    """
    Performs object detection on selected frames using YOLOv8.
    
    Args:
        frames (List[str]): List of frame file paths.
        model_path (str): Path to the YOLO model.
        output_dir (str): Directory to save detection results.
        
    Returns:
        List[Optional[dict]]: List of detection results for each frame, None for failed detections.
        
    Raises:
        ValueError: If frames list is empty or model_path doesn't exist.
    """
    if not frames:
        raise ValueError("Empty frames list provided")
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found at: {model_path}")

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    results_list = []
    
    total_frames = len(frames)
    for idx, frame_path in enumerate(frames, 1):
        try:
            if not os.path.exists(frame_path):
                print(f"Warning: Frame not found: {frame_path}")
                results_list.append(None)
                continue
                
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Failed to read frame: {frame_path}")
                results_list.append(None)
                continue
                
            results = model(frame)
            results.save(save_dir=output_dir)
            results_list.append(results[0].boxes.data.cpu().numpy())
            
            print(f"Processed frame {idx}/{total_frames}")
            
        except Exception as e:
            print(f"Error processing frame {frame_path}: {str(e)}")
            results_list.append(None)
            
    return results_list
