# Riesz Pyramid Visualization

This project implements a Riesz pyramid visualization tool for video processing. It applies the Riesz transform to create a multi-scale representation of video frames, allowing for advanced analysis of motion and orientation in the video.

## Features

- Load frames from a video file
- Compute Riesz pyramids for each frame
- Visualize the Riesz pyramid components (Rx, Ry, and magnitude)
- Generate an output video showing the Riesz pyramid visualization for each frame

## Requirements

- Python 3.x
- NumPy
- SciPy
- OpenCV (cv2)
- Matplotlib

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install numpy scipy opencv-python matplotlib
```

## Usage

1. Place your input video file in the project directory
2. Update the `video_path` variable in `main.py` with your video file name
3. Run the script:

```bash
python main.py
```

4. The script will generate an output video file named `riesz_pyramid_visualization.mp4`

## How it works

1. The script loads frames from the input video
2. For each frame, it computes a Riesz pyramid with a specified number of levels
3. The Riesz pyramid components (Rx, Ry, and magnitude) are visualized for each level
4. The visualizations are compiled into an output video

This tool can be useful for analyzing motion, texture, and orientation information in videos across different scales.
