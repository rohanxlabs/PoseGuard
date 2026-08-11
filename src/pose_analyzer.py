import numpy as np
from typing import Dict, Optional, Tuple
import yaml


def _vec(a, b):
    """Vector from point a to point b."""
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)


def angle_between(p1, vertex, p2) -> Optional[float]:
    """
    Compute angle at `vertex` formed by rays to p1 and p2.
    Returns degrees [0, 180].
    """
    if any(p is None for p in [p1, vertex, p2]):
        return None
    v1 = _vec(vertex[:2], p1[:2])
    v2 = _vec(vertex[:2], p2[:2])
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def shin_from_vertical_angle(knee, ankle) -> Optional[float]:
    """
    Ankle inversion proxy: angle of the shin vector (knee → ankle)
    from downward vertical [0, 1].  A large lateral tilt suggests
    inversion stress.  Returns degrees [0, 90].
    """
    if knee is None or ankle is None:
        return None
    shin = _vec(knee[:2], ankle[:2])
    norm = np.linalg.norm(shin)
    if norm == 0:
        return None
    vertical_down = np.array([0.0, 1.0], dtype=np.float32)
    cos_a = np.clip(np.dot(shin / norm, vertical_down), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def joint_velocity(kp_current: Dict, kp_previous: Dict, joint: str) -> Optional[float]:
    """Euclidean pixel displacement per frame for a joint."""
    if kp_current.get(joint) is None or kp_previous.get(joint) is None:
        return None
    c, p = kp_current[joint], kp_previous[joint]
    return float(np.linalg.norm(np.array(c[:2]) - np.array(p[:2])))


class PoseAnalyzer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg["injury"]

    def compute_angles(self, kps: Dict) -> Dict[str, Optional[float]]:
        """Compute all biomechanically relevant joint angles."""
        kp = kps  # shorthand

        angles = {
            # Lower body
            "left_knee":    angle_between(kp.get("left_hip"),    kp.get("left_knee"),   kp.get("left_ankle")),
            "right_knee":   angle_between(kp.get("right_hip"),   kp.get("right_knee"),  kp.get("right_ankle")),
            "left_hip":     angle_between(kp.get("left_shoulder"), kp.get("left_hip"),  kp.get("left_knee")),
            "right_hip":    angle_between(kp.get("right_shoulder"), kp.get("right_hip"), kp.get("right_knee")),
            # Ankle inversion proxy: shin-to-vertical tilt angle (no third point needed)
            "left_ankle":   shin_from_vertical_angle(kp.get("left_knee"),   kp.get("left_ankle")),
            "right_ankle":  shin_from_vertical_angle(kp.get("right_knee"),  kp.get("right_ankle")),

            # Upper body
            "left_elbow":   angle_between(kp.get("left_shoulder"),  kp.get("left_elbow"),  kp.get("left_wrist")),
            "right_elbow":  angle_between(kp.get("right_shoulder"), kp.get("right_elbow"), kp.get("right_wrist")),
            "left_shoulder": angle_between(kp.get("left_elbow"),   kp.get("left_shoulder"), kp.get("left_hip")),
            "right_shoulder": angle_between(kp.get("right_elbow"), kp.get("right_shoulder"), kp.get("right_hip")),

            # Spine / torso
            "spine_lateral": self._spine_lateral_angle(kps),
            "trunk_forward":  self._trunk_forward_angle(kps),
        }
        return angles

    def _spine_lateral_angle(self, kps: Dict) -> Optional[float]:
        """Angle of midpoint-shoulder to midpoint-hip vector from vertical."""
        ls, rs = kps.get("left_shoulder"), kps.get("right_shoulder")
        lh, rh = kps.get("left_hip"), kps.get("right_hip")
        if any(x is None for x in [ls, rs, lh, rh]):
            return None
        shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        v = _vec(hip_mid, shoulder_mid)
        vertical = np.array([0, -1], dtype=np.float32)
        cos_a = np.clip(np.dot(v, vertical) / (np.linalg.norm(v) + 1e-6), -1, 1)
        return float(np.degrees(np.arccos(cos_a)))

    def _trunk_forward_angle(self, kps: Dict) -> Optional[float]:
        """Forward lean angle using shoulder and hip midpoints."""
        ls, rs = kps.get("left_shoulder"), kps.get("right_shoulder")
        lh, rh = kps.get("left_hip"), kps.get("right_hip")
        if any(x is None for x in [ls, rs, lh, rh]):
            return None
        shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
        hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
        torso = shoulder_mid - hip_mid
        vertical = np.array([0, -1], dtype=np.float32)
        cos_a = np.clip(np.dot(torso, vertical) / (np.linalg.norm(torso) + 1e-6), -1, 1)
        return float(np.degrees(np.arccos(cos_a)))

    def limb_asymmetry(self, kps: Dict) -> Dict[str, Optional[float]]:
        """
        Compare left vs right limb segment lengths.
        Asymmetry > threshold may indicate guarding or injury compensation.
        """
        def seg_len(a, b):
            if kps.get(a) is None or kps.get(b) is None:
                return None
            return np.linalg.norm(np.array(kps[a][:2]) - np.array(kps[b][:2]))

        left_thigh  = seg_len("left_hip",  "left_knee")
        right_thigh = seg_len("right_hip", "right_knee")
        left_shin   = seg_len("left_knee", "left_ankle")
        right_shin  = seg_len("right_knee", "right_ankle")
        left_upper_arm  = seg_len("left_shoulder",  "left_elbow")
        right_upper_arm = seg_len("right_shoulder", "right_elbow")

        def asym_ratio(l, r):
            if l is None or r is None or (l + r) == 0:
                return None
            return abs(l - r) / ((l + r) / 2)

        return {
            "thigh_asymmetry": asym_ratio(left_thigh, right_thigh),
            "shin_asymmetry":  asym_ratio(left_shin, right_shin),
            "arm_asymmetry":   asym_ratio(left_upper_arm, right_upper_arm),
        }