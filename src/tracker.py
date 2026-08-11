import numpy as np
from typing import List, Dict, Optional
from scipy.optimize import linear_sum_assignment


def iou(boxA, boxB) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0


class Track:
    _id_counter = 0

    @classmethod
    def reset_counter(cls):
        """Reset global ID counter (useful for tests)."""
        cls._id_counter = 0

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
    Assigns consistent track IDs across frames using Hungarian matching.
    ID assignment uses a direct index map — not float bbox equality — to
    avoid ambiguity when multiple detections share similar coordinates.
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
        # Age all existing tracks
        for t in self.tracks:
            t.mark_missed()

        # Remove detections that report track_id from a prior call
        for d in detections:
            d["track_id"] = None

        if not detections:
            # Remove expired tracks
            expired_ids = {t.id for t in self.tracks
                           if t.frames_since_update > self.max_age}
            self.tracks = [t for t in self.tracks
                           if t.frames_since_update <= self.max_age]
            return [], expired_ids

        det_boxes = [d["bbox"] for d in detections]

        # det_index_to_track_id: maps detection index → assigned track id
        det_index_to_track_id: Dict[int, int] = {}

        if not self.tracks:
            for j, d in enumerate(detections):
                new_track = Track(d["bbox"], d)
                self.tracks.append(new_track)
                if new_track.hits >= self.min_hits:
                    det_index_to_track_id[j] = new_track.id
        else:
            track_boxes = [t.bbox for t in self.tracks]

            # Build IoU cost matrix
            cost = np.zeros((len(self.tracks), len(det_boxes)))
            for i, tb in enumerate(track_boxes):
                for j, db in enumerate(det_boxes):
                    cost[i, j] = 1.0 - iou(tb, db)

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < (1.0 - self.iou_thresh):
                    self.tracks[r].update(det_boxes[c], detections[c])
                    matched_dets.add(c)
                    # Record track id directly — no bbox re-scan
                    if self.tracks[r].hits >= self.min_hits:
                        det_index_to_track_id[c] = self.tracks[r].id

            # New tracks for unmatched detections
            for j, d in enumerate(detections):
                if j not in matched_dets:
                    new_track = Track(d["bbox"], d)
                    self.tracks.append(new_track)
                    if new_track.hits >= self.min_hits:
                        det_index_to_track_id[j] = new_track.id

        # Remove stale tracks
        expired_ids = {t.id for t in self.tracks
                       if t.frames_since_update > self.max_age}
        self.tracks = [t for t in self.tracks
                       if t.frames_since_update <= self.max_age]

        # Assign track IDs to detections using the direct map
        for j, d in enumerate(detections):
            d["track_id"] = det_index_to_track_id.get(j, -1)

        return detections, expired_ids

    def update_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Compatibility wrapper that discards the expired_ids set
        for callers that don't need it.
        """
        result, _ = self.update(detections)
        return result
