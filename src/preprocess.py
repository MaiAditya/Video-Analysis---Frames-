# src/preprocess.py
import cv2
import os
from typing import List

def extract_frames(video_path: str, output_dir: str, frame_rate: int = 15) -> List[str]:
    """
    Extracts frames from a video at the specified frame rate.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the extracted frames.
        frame_rate (int): Number of frames to extract per second.
        
    Returns:
        List[str]: List of paths to the extracted frames.
        
    Raises:
        ValueError: If video_path doesn't exist or frame_rate is invalid.
        RuntimeError: If video cannot be opened or processed.
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
    if frame_rate <= 0:
        raise ValueError(f"Frame rate must be positive, got {frame_rate}")
        
    os.makedirs(output_dir, exist_ok=True)
    extracted_frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = fps // frame_rate
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
                saved_count += 1
                
                # Print progress every 10% of total frames
                if total_frames > 0 and frame_count % (total_frames // 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}%")
            
            frame_count += 1
            
    except Exception as e:
        raise RuntimeError(f"Error processing video: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
            
    print(f"Extracted {saved_count} frames and saved to {output_dir}")
    return extracted_frames
