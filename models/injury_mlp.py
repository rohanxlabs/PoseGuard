"""
Optional: Train a lightweight MLP on extracted pose features
instead of (or in addition to) the rule-based classifier.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict


def build_feature_vector(angles: Dict, asymmetry: Dict, velocities: Dict) -> np.ndarray:
    """
    Flatten all analysis outputs into a fixed-length feature vector.
    """
    angle_keys = [
        "left_knee", "right_knee", "left_hip", "right_hip",
        "left_ankle", "right_ankle", "left_elbow", "right_elbow",
        "left_shoulder", "right_shoulder", "spine_lateral", "trunk_forward"
    ]
    asym_keys = ["thigh_asymmetry", "shin_asymmetry", "arm_asymmetry"]
    vel_keys  = ["left_knee", "right_knee", "left_ankle", "right_ankle",
                 "left_wrist", "right_wrist"]

    feats = []
    for k in angle_keys:
        v = angles.get(k)
        feats.append(v if v is not None else 0.0)
    for k in asym_keys:
        v = asymmetry.get(k)
        feats.append(v if v is not None else 0.0)
    for k in vel_keys:
        v = velocities.get(k)
        feats.append(v if v is not None else 0.0)

    return np.array(feats, dtype=np.float32)  # (21,)


class InjuryMLP(nn.Module):
    """
    Binary classifier: injury risk (0=safe, 1=at-risk).
    Input dim: 21 (12 angles + 3 asymmetry + 6 velocities)
    """
    def __init__(self, input_dim: int = 21, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, feat_vec: np.ndarray) -> float:
        """Returns probability of at-risk class."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(feat_vec).unsqueeze(0)
            logits = self.forward(x)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return prob