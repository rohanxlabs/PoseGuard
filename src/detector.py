import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import yaml


# YOLOv8-Pose keypoint indices (COCO 17-point skeleton)
KEYPOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]


class PoseDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.model = YOLO(self.cfg["model"]["weights"])
        self.conf = self.cfg["model"]["confidence"]
        self.iou = self.cfg["model"]["iou_threshold"]
        self.device = self.cfg["model"]["device"]
        self.kp_conf_thresh = self.cfg["pose"]["keypoint_confidence_threshold"]

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8-pose on a single frame.
        Returns list of person dicts with bbox, keypoints, confidence.
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )

        persons = []
        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
            boxes = result.boxes.xyxy.cpu().numpy()                 # (N, 4)
            scores = result.boxes.conf.cpu().numpy()                # (N,)

            for i in range(len(boxes)):
                kps = keypoints_data[i]   # (17, 3) → x, y, conf

                # Filter low-confidence keypoints
                valid_kps = {}
                for name, idx in KEYPOINTS.items():
                    x, y, c = kps[idx]
                    if c >= self.kp_conf_thresh:
                        valid_kps[name] = (float(x), float(y), float(c))
                    else:
                        valid_kps[name] = None

                persons.append({
                    "bbox": boxes[i].tolist(),          # [x1, y1, x2, y2]
                    "score": float(scores[i]),
                    "keypoints": valid_kps,
                    "keypoints_raw": kps,               # raw (17,3) array
                    "track_id": None,                   # filled by tracker
                })

        return persons