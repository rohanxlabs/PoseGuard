import numpy as np
from typing import Dict, Optional, Deque
from collections import deque, defaultdict


class TemporalAnalyzer:
    """
    Maintains per-track keypoint history for:
    - Velocity & acceleration computation
    - Smooth angle trends
    - Injury event deduplication (debounce)
    """
    def __init__(self, window: int = 5, debounce_frames: int = 15):
        self.window = window
        self.debounce_frames = debounce_frames
        self._kp_history: Dict[int, Deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_event_frame: Dict[str, int] = {}  # (track_id, injury_type) → frame

    def update(self, track_id: int, keypoints: Dict):
        self._kp_history[track_id].append(keypoints)

    def get_velocities(self, track_id: int) -> Dict[str, Optional[float]]:
        """Per-joint velocity (pixels/frame) using last 2 frames."""
        hist = self._kp_history[track_id]
        if len(hist) < 2:
            return {}
        curr, prev = hist[-1], hist[-2]
        velocities = {}
        for joint in curr:
            if curr[joint] is not None and prev.get(joint) is not None:
                velocities[joint] = float(np.linalg.norm(
                    np.array(curr[joint][:2]) - np.array(prev[joint][:2])
                ))
            else:
                velocities[joint] = None
        return velocities

    def get_smoothed_keypoints(self, track_id: int) -> Optional[Dict]:
        """Return keypoint positions averaged over history window."""
        hist = list(self._kp_history[track_id])
        if not hist:
            return None
        smoothed = {}
        joints = hist[-1].keys()
        for joint in joints:
            vals = [h[joint] for h in hist if h.get(joint) is not None]
            if vals:
                avg_x = np.mean([v[0] for v in vals])
                avg_y = np.mean([v[1] for v in vals])
                avg_c = np.mean([v[2] for v in vals])
                smoothed[joint] = (avg_x, avg_y, avg_c)
            else:
                smoothed[joint] = None
        return smoothed

    def should_emit_event(self, track_id: int, injury_type: str, frame_idx: int) -> bool:
        """Debounce: suppress repeated events for same injury type within window."""
        key = f"{track_id}_{injury_type}"
        last = self._last_event_frame.get(key, -999)
        if frame_idx - last >= self.debounce_frames:
            self._last_event_frame[key] = frame_idx
            return True
        return False

    def reset_track(self, track_id: int):
        """
        Clear all temporal state for a track that has expired.
        Prevents stale history bleeding into a re-used track ID.
        """
        self._kp_history.pop(track_id, None)
        # Remove debounce keys for this track
        keys_to_remove = [k for k in self._last_event_frame if k.startswith(f"{track_id}_")]
        for k in keys_to_remove:
            del self._last_event_frame[k]