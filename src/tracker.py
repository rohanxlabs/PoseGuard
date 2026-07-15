import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment


def iou(boxA, boxB) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0


class Track:
    _id_counter = 0

    def __init__(self, bbox: List[float], person_data: Dict):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = bbox
        self.person_data = person_data
        self.hits = 1
        self.age = 0
        self.frames_since_update = 0

    def update(self, bbox: List[float], person_data: Dict):
        self.bbox = bbox
        self.person_data = person_data
        self.hits += 1
        self.frames_since_update = 0

    def mark_missed(self):
        self.frames_since_update += 1
        self.age += 1


class SortTracker:
    """
    Lightweight SORT-style IoU tracker (no Kalman filter dependency).
    Assigns consistent track IDs across frames.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        tc = cfg["tracking"]
        self.max_age = tc["max_age"]
        self.min_hits = tc["min_hits"]
        self.iou_thresh = tc["iou_threshold"]
        self.tracks: List[Track] = []

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Match current detections to existing tracks.
        Returns detections with `track_id` filled in.
        """
        # Age all tracks
        for t in self.tracks:
            t.mark_missed()

        if not detections:
            self.tracks = [t for t in self.tracks if t.frames_since_update <= self.max_age]
            return []

        det_boxes = [d["bbox"] for d in detections]

        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(d["bbox"], d))
        else:
            track_boxes = [t.bbox for t in self.tracks]

            # Build IoU cost matrix
            cost = np.zeros((len(self.tracks), len(det_boxes)))
            for i, tb in enumerate(track_boxes):
                for j, db in enumerate(det_boxes):
                    cost[i, j] = 1 - iou(tb, db)

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1 - self.iou_thresh):
                    self.tracks[r].update(det_boxes[c], detections[c])
                    matched_dets.add(c)

            # New tracks for unmatched detections
            for j, d in enumerate(detections):
                if j not in matched_dets:
                    self.tracks.append(Track(d["bbox"], d))

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.frames_since_update <= self.max_age]

        # Assign IDs back to detections
        for d in detections:
            for t in self.tracks:
                if t.bbox == d["bbox"] and t.hits >= self.min_hits:
                    d["track_id"] = t.id
                    break
            if d["track_id"] is None:
                d["track_id"] = -1

        return detections