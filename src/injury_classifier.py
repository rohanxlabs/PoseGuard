import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
from collections import deque


INJURY_TYPES = {
    "ACL_RISK":        "ACL / Knee Ligament Risk",
    "ANKLE_SPRAIN":    "Ankle Sprain Risk",
    "HAMSTRING":       "Hamstring Strain Risk",
    "FALL":            "Fall / Impact Detected",
    "SHOULDER":        "Shoulder Injury Risk",
    "SPINE":           "Spine / Back Stress",
    "ELBOW":           "Elbow Hyperextension Risk",
    "ASYMMETRY":       "Compensatory Asymmetry",
    "NONE":            "No Injury Detected",
}


class InjuryEvent:
    def __init__(self, injury_type: str, risk_score: float, frame_idx: int,
                 track_id: Optional[int], details: str):
        self.injury_type = injury_type
        self.risk_score = risk_score
        self.frame_idx = frame_idx
        self.track_id = track_id
        self.details = details
        self.label = INJURY_TYPES.get(injury_type, injury_type)

    def __repr__(self):
        return (f"[Frame {self.frame_idx}] Track {self.track_id} | "
                f"{self.label} | Score: {self.risk_score:.2f} | {self.details}")


class InjuryClassifier:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg["injury"]
        self.high_thresh = self.cfg["high_risk_score"]
        self.med_thresh = self.cfg["medium_risk_score"]

        # Per-track velocity history for spike detection
        self._velocity_history: Dict[int, deque] = {}
        self._position_history: Dict[int, deque] = {}
        self._fall_history: Dict[int, deque] = {}

    def classify(
        self,
        angles: Dict[str, Optional[float]],
        asymmetry: Dict[str, Optional[float]],
        keypoints: Dict,
        velocities: Dict[str, Optional[float]],
        track_id: int,
        frame_idx: int,
    ) -> List[InjuryEvent]:
        """
        Run all injury detection rules.
        Returns list of InjuryEvent objects (can be multiple per person per frame).
        """
        events = []

        # --- Rule 1: Knee hyperextension → ACL risk ---
        for side in ["left", "right"]:
            angle = angles.get(f"{side}_knee")
            if angle is not None and angle > self.cfg["knee_hyperextension_angle"]:
                score = min(1.0, (angle - self.cfg["knee_hyperextension_angle"]) / 15 * 0.8 + 0.5)
                events.append(InjuryEvent(
                    "ACL_RISK", score, frame_idx, track_id,
                    f"{side.capitalize()} knee angle: {angle:.1f}°"
                ))

        # --- Rule 2: Ankle inversion ---
        for side in ["left", "right"]:
            angle = angles.get(f"{side}_ankle")
            if angle is not None and angle > self.cfg["ankle_inversion_angle"]:
                score = min(1.0, (angle - self.cfg["ankle_inversion_angle"]) / 20 * 0.7 + 0.4)
                events.append(InjuryEvent(
                    "ANKLE_SPRAIN", score, frame_idx, track_id,
                    f"{side.capitalize()} ankle angle: {angle:.1f}°"
                ))

        # --- Rule 3: Elbow hyperextension ---
        for side in ["left", "right"]:
            angle = angles.get(f"{side}_elbow")
            if angle is not None and angle > self.cfg["elbow_hyperextension_angle"]:
                score = min(1.0, (angle - self.cfg["elbow_hyperextension_angle"]) / 10 * 0.7 + 0.5)
                events.append(InjuryEvent(
                    "ELBOW", score, frame_idx, track_id,
                    f"{side.capitalize()} elbow hyperextension: {angle:.1f}°"
                ))

        # --- Rule 4: Spine lateral flexion ---
        spine_angle = angles.get("spine_lateral")
        if spine_angle is not None and spine_angle > self.cfg["spine_lateral_flexion_angle"]:
            score = min(1.0, (spine_angle - self.cfg["spine_lateral_flexion_angle"]) / 25 * 0.65 + 0.4)
            events.append(InjuryEvent(
                "SPINE", score, frame_idx, track_id,
                f"Spine lateral angle: {spine_angle:.1f}°"
            ))

        # --- Rule 5: Hip abduction ---
        for side in ["left", "right"]:
            angle = angles.get(f"{side}_hip")
            if angle is not None and angle > self.cfg["hip_abduction_angle"]:
                score = min(1.0, (angle - self.cfg["hip_abduction_angle"]) / 20 * 0.6 + 0.4)
                events.append(InjuryEvent(
                    "HAMSTRING", score, frame_idx, track_id,
                    f"{side.capitalize()} hip abduction: {angle:.1f}°"
                ))

        # --- Rule 6: Velocity spike ---
        self._update_velocity_history(track_id, velocities)
        spike_joint = self._detect_velocity_spike(track_id)
        if spike_joint:
            events.append(InjuryEvent(
                "FALL", 0.70, frame_idx, track_id,
                f"Sudden velocity spike at: {spike_joint}"
            ))

        # --- Rule 7: Fall detection (rapid torso drop) ---
        fall_detected = self._detect_fall(track_id, keypoints)
        if fall_detected:
            events.append(InjuryEvent(
                "FALL", 0.85, frame_idx, track_id,
                "Rapid torso descent / fall detected"
            ))

        # --- Rule 8: Limb asymmetry ---
        for key, ratio in asymmetry.items():
            if ratio is not None and ratio > self.cfg["asymmetry_threshold"]:
                events.append(InjuryEvent(
                    "ASYMMETRY", min(1.0, ratio * 0.8), frame_idx, track_id,
                    f"{key.replace('_', ' ').title()}: {ratio:.2f}"
                ))

        return events

    def _update_velocity_history(self, track_id: int, velocities: Dict):
        if track_id not in self._velocity_history:
            self._velocity_history[track_id] = deque(maxlen=10)
        self._velocity_history[track_id].append(velocities)

    def _detect_velocity_spike(self, track_id: int) -> Optional[str]:
        hist = self._velocity_history.get(track_id)
        if hist is None or len(hist) < 3:
            return None
        for joint in ["left_knee", "right_knee", "left_ankle", "right_ankle"]:
            recent = [h.get(joint) for h in list(hist)[-3:]]
            recent = [v for v in recent if v is not None]
            if len(recent) >= 2 and max(recent) > self.cfg["velocity_spike_threshold"]:
                return joint
        return None

    def _detect_fall(self, track_id: int, keypoints: Dict) -> bool:
        """Detect rapid vertical drop of hip/shoulder midpoints."""
        ls = keypoints.get("left_shoulder")
        rs = keypoints.get("right_shoulder")
        lh = keypoints.get("left_hip")
        rh = keypoints.get("right_hip")

        if any(x is None for x in [ls, rs, lh, rh]):
            return False

        torso_y = (ls[1] + rs[1] + lh[1] + rh[1]) / 4  # avg Y (down = larger)

        if track_id not in self._fall_history:
            self._fall_history[track_id] = deque(maxlen=self.cfg["fall_detection_frames"])
        self._fall_history[track_id].append(torso_y)

        hist = self._fall_history[track_id]
        if len(hist) == self.cfg["fall_detection_frames"]:
            delta = hist[-1] - hist[0]
            if delta > self.cfg["acceleration_threshold"]:
                return True
        return False

    def get_risk_level(self, score: float) -> Tuple[str, Tuple[int, int, int]]:
        if score >= self.high_thresh:
            return "HIGH",   (0, 0, 220)     # BGR red
        elif score >= self.med_thresh:
            return "MEDIUM", (0, 140, 255)   # BGR orange
        else:
            return "LOW",    (0, 220, 100)   # BGR green