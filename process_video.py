import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_tracker import FeatureTracker


def process_video(video_path, output_path='output.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return []

    tracker = FeatureTracker()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps} FPS")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame")
        return []

    prev_kp, prev_desc = tracker.detect_features(prev_frame)

    frame_count = 0
    motion_history = []
    cum_x, cum_y = 0.0, 0.0

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_kp, curr_desc = tracker.detect_features(curr_frame)
        motion = None

        if prev_desc is not None and curr_desc is not None:
            matches = tracker.match_features(prev_desc, curr_desc)
            motion = tracker.estimate_motion(prev_kp, curr_kp, matches)

            if motion:
                cum_x += motion['dx']
                cum_y += motion['dy']
                motion['cum_x'] = cum_x
                motion['cum_y'] = cum_y

                cv2.putText(curr_frame, f"Features: {len(curr_kp)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(curr_frame, f"Matches: {len(matches)}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(curr_frame, f"Motion: dx={motion['dx']:.1f}, dy={motion['dy']:.1f}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        motion_history.append(motion or {'dx': 0, 'dy': 0, 'inliers': 0, 'cum_x': cum_x, 'cum_y': cum_y})

        result = tracker.draw_features(curr_frame, curr_kp)
        out.write(result)

        prev_frame = curr_frame
        prev_kp = curr_kp
        prev_desc = curr_desc

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    return motion_history


def plot_motion(motion_history, output_file='motion_plot.png'):
    dx = [m['dx'] for m in motion_history]
    dy = [m['dy'] for m in motion_history]
    cum_x = [m['cum_x'] for m in motion_history]
    cum_y = [m['cum_y'] for m in motion_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(dx, label='X', color='steelblue')
    ax1.plot(dy, label='Y', color='darkorange')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Motion (pixels)')
    ax1.set_title('Per-frame Delta')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(cum_x, label='X', color='steelblue')
    ax2.plot(cum_y, label='Y', color='darkorange')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Cumulative (pixels)')
    ax2.set_title('Camera Trajectory')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")


if __name__ == '__main__':
    print("Processing video...")
    motion = process_video('drone_footage.mp4', 'tracked_output.mp4')
    print(f"Processed {len(motion)} frames")

    if motion:
        plot_motion(motion)