import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.injury_classifier import InjuryEvent, INJURY_TYPES
from src.detector import SKELETON_PAIRS, KEYPOINTS


RISK_COLORS = {
    "HIGH":   (0, 0, 220),
    "MEDIUM": (0, 140, 255),
    "LOW":    (0, 220, 100),
    "NONE":   (200, 200, 200),
}


class Visualizer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_skeleton(self, frame: np.ndarray, person: Dict,
                      color: Tuple = (0, 255, 120), thickness: int = 2) -> np.ndarray:
        kps = person["keypoints"]
        kp_list = [kps.get(name) for name in KEYPOINTS.keys()]  # indexed list

        # Draw limb connections
        for (i, j) in SKELETON_PAIRS:
            if kp_list[i] is not None and kp_list[j] is not None:
                pt1 = (int(kp_list[i][0]), int(kp_list[i][1]))
                pt2 = (int(kp_list[j][0]), int(kp_list[j][1]))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        # Draw keypoints
        for kp in kp_list:
            if kp is not None:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255, 255, 255), -1)
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, 1)

        return frame

    def draw_bbox(self, frame: np.ndarray, person: Dict,
                  color: Tuple, track_id: Optional[int]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        if track_id is not None and track_id != -1:
            label = f"ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 8), self.font, 0.55, color, 2, cv2.LINE_AA)
        return frame

    def draw_injury_alerts(self, frame: np.ndarray,
                           events: List[InjuryEvent],
                           person: Dict) -> np.ndarray:
        if not events:
            return frame

        # Annotate at top of bounding box
        x1, y1 = int(person["bbox"][0]), int(person["bbox"][1])

        for idx, event in enumerate(events[:3]):  # show max 3 events per person
            score = event.risk_score
            level = "HIGH" if score >= 0.75 else ("MEDIUM" if score >= 0.45 else "LOW")
            color = RISK_COLORS[level]
            label = f"{event.label} [{level}] {score:.0%}"

            y_pos = y1 - 30 - idx * 22
            # Background pill
            (w, h), _ = cv2.getTextSize(label, self.font, 0.48, 1)
            cv2.rectangle(frame, (x1 - 2, y_pos - h - 4), (x1 + w + 4, y_pos + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y_pos), self.font, 0.48, color, 1, cv2.LINE_AA)

        return frame

    def draw_hud(self, frame: np.ndarray, frame_idx: int,
                 total_events: int, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]

        # Top-left HUD box
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "POSEGUARD", (8, 18),
                    self.font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_idx:06d}  |  FPS: {fps:.1f}", (8, 38),
                    self.font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Total Events: {total_events}", (8, 58),
                    self.font, 0.45, (0, 220, 100) if total_events == 0 else (0, 100, 255),
                    1, cv2.LINE_AA)
        return frame

    def draw_angles(self, frame: np.ndarray, person: Dict,
                    angles: Dict, joints_to_show: List[str] = None) -> np.ndarray:
        """Optionally overlay joint angles on skeleton."""
        kps = person["keypoints"]
        joints_to_show = joints_to_show or ["left_knee", "right_knee", "left_elbow", "right_elbow"]
        joint_kp_map = {
            "left_knee": "left_knee", "right_knee": "right_knee",
            "left_elbow": "left_elbow", "right_elbow": "right_elbow",
            "left_hip": "left_hip", "right_hip": "right_hip",
        }
        for joint in joints_to_show:
            angle = angles.get(joint)
            kp_name = joint_kp_map.get(joint)
            if angle is not None and kp_name and kps.get(kp_name) is not None:
                x, y = int(kps[kp_name][0]) + 8, int(kps[kp_name][1])
                cv2.putText(frame, f"{angle:.0f}°", (x, y), self.font,
                            0.42, (255, 230, 100), 1, cv2.LINE_AA)
        return frame