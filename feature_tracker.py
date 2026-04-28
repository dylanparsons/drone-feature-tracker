import cv2
import numpy as np


class FeatureTracker:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def detect_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mask = (np.abs(laplacian) > 5).astype(np.uint8) * 255
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)
        return keypoints, descriptors

    def draw_features(self, frame, keypoints):
        return cv2.drawKeypoints(
            frame, keypoints, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    def match_features(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []
        matches = self.bf.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return sorted(good, key=lambda x: x.distance)

    def estimate_motion(self, kp1, kp2, matches):
        if len(matches) < 10:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]])

        M, mask = cv2.estimateAffinePartial2D(pts1, pts2)
        if M is not None:
            return {'dx': M[0, 2], 'dy': M[1, 2], 'inliers': int(mask.sum())}

        return None


def main():
    cap = cv2.VideoCapture('drone_footage.mp4')

    if not cap.isOpened():
        print("ERROR: Could not open drone_footage.mp4")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame")
        return

    print(f"Frame shape: {frame.shape}")

    tracker = FeatureTracker()
    kp, desc = tracker.detect_features(frame)

    result = tracker.draw_features(frame, kp)
    cv2.imwrite('features_detected.jpg', result)

    print(f"Detected {len(kp)} features")
    cap.release()


if __name__ == '__main__':
    main()