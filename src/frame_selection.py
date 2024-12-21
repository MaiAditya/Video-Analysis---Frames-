# src/frame_selection.py
import cv2
import numpy as np
import os

def uniform_sampling(frames, num_samples=20):
    """
    Selects frames uniformly across the sequence.
    
    Args:
        frames (list): List of frame file paths.
        num_samples (int): Number of frames to sample.
        
    Returns:
        list: Selected frame file paths.
    """
    if not frames:
        raise ValueError("Empty frames list provided")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_samples > len(frames):
        return frames  # Return all frames if num_samples is larger than available frames
    step = len(frames) // num_samples
    return frames[::step][:num_samples]

def scene_change_detection(frames, threshold=30.0):
    """
    Selects frames based on significant scene changes using color histogram comparison.
    
    Args:
        frames (list): List of frame file paths.
        threshold (float): Threshold for histogram difference. Higher values mean
            fewer frames will be selected. Typical values range from 20.0 to 50.0.
        
    Returns:
        list: Selected frame file paths.
        
    Raises:
        ValueError: If frames list is empty.
        
    Example:
        >>> frames = ['frame1.jpg', 'frame2.jpg', 'frame3.jpg']
        >>> selected = scene_change_detection(frames, threshold=25.0)
    """
    if not frames:
        raise ValueError("Empty frames list provided")
    
    selected_frames = [frames[0]]
    prev_hist = None
    
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        # Calculate histogram for each channel
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        
        # Normalize and concatenate
        hist = np.concatenate([
            cv2.normalize(hist_b, hist_b).flatten(),
            cv2.normalize(hist_g, hist_g).flatten(),
            cv2.normalize(hist_r, hist_r).flatten()
        ])
        
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > threshold:
                selected_frames.append(frame_path)
        prev_hist = hist
    
    return selected_frames

def motion_based_sampling(frames, threshold=0.2):
    """
    Selects frames with significant motion using optical flow.
    
    Args:
        frames (list): List of frame file paths.
        threshold (float): Motion magnitude threshold.
        
    Returns:
        list: Selected frame file paths.
    """
    if not frames:
        raise ValueError("Empty frames list provided")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
        
    selected_frames = []
    prev_frame = None
    
    for frame_path in frames:
        try:
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
                
            if prev_frame is not None:
                if prev_frame.shape != frame.shape:
                    continue
                    
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                motion_magnitude = np.linalg.norm(flow, axis=2).mean()
                if motion_magnitude > threshold:
                    selected_frames.append(frame_path)
            prev_frame = frame
        except Exception as e:
            print(f"Error processing frame {frame_path}: {str(e)}")
            continue
    
    return selected_frames
