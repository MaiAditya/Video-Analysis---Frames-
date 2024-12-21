# Video Analysis for Shoppable Items

A Python-based solution for detecting shoppable items in video content using intelligent frame selection and object detection.

## Overview

This project provides a pipeline for analyzing video content to detect potential shoppable items. It employs a three-stage approach:

1. Frame Extraction
2. Intelligent Frame Selection
3. Object Detection

## Technical Architecture

### 1. Frame Extraction (`src/preprocess.py`)
- Extracts frames from input videos at configurable frame rates
- Handles various video formats using OpenCV
- Implements error handling for corrupt or invalid video files
- Outputs individual frames as images for further processing

### 2. Frame Selection (`src/frame_selection.py`)
Implements two strategies for intelligent frame selection:

a) Uniform Sampling
- Selects frames at regular intervals
- Configurable sample size
- Ensures even coverage across video duration

b) Scene Change Detection
- Uses color histogram comparison
- Detects significant visual changes between frames
- Adjustable threshold for sensitivity control
- Reduces redundancy in processed frames

### 3. Object Detection (`src/object_detection.py`)
- Utilizes YOLOv8 for object detection
- Processes selected frames to identify potential products
- Saves detection results with bounding boxes
- Provides detection confidence scores

## Requirements

- Python 3.9 or higher
- Dependencies managed via Poetry:
  - PyTorch 2.1.0+
  - OpenCV 4.8.0+
  - Ultralytics 8.0.0+
  - Additional dependencies listed in pyproject.toml

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MaiAditya/Video-Analysis---Frames-.git
cd video-analysis-assignment
```
