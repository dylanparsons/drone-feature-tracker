# drone-feature-tracker

ORB-based feature tracking pipeline for estimating camera motion in drone footage.

## How it works

1. Detect ORB keypoints with a Laplacian variance mask to filter water/sky regions
2. Match across frames using Lowe's ratio test (0.75)
3. Estimate (dx, dy) via RANSAC partial affine transform
4. Output annotated video + motion plot

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Single frame test
python feature_tracker.py

# Full video
python process_video.py
```

Outputs: `features_detected.jpg`, `tracked_output.mp4`, `motion_plot.png`