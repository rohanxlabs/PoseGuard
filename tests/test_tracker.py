"""
Test multi-person tracking with IoU matching.
"""
import pytest
from src.tracker import SortTracker, Track, iou


class TestIoU:
    """Test Intersection over Union calculation."""

    def test_perfect_overlap(self):
        """Identical boxes should have IoU = 1.0."""
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        assert iou(box1, box2) == 1.0

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        box1 = [10, 10, 50, 50]
        box2 = [60, 60, 100, 100]
        assert iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        box1 = [10, 10, 50, 50]
        box2 = [30, 30, 70, 70]
        result = iou(box1, box2)
        assert 0 < result < 1
        # Area of overlap: 20x20 = 400
        # Area of union: (40*40) + (40*40) - 400 = 2800
        # IoU = 400 / 2800 ≈ 0.143
        assert 0.14 < result < 0.15


class TestTrackerInitialization:
    """Test tracker setup and configuration."""

    def test_tracker_creates_without_config(self):
        """Tracker should initialize with default config."""
        # This requires a valid config.yaml to exist
        tracker = SortTracker("configs/config.yaml")
        assert tracker is not None
        assert len(tracker.tracks) == 0


class TestTrackCreation:
    """Test new track creation and ID assignment."""

    def test_first_detection_creates_track(self):
        """First detection should create a new track."""
        Track.reset_counter()
        tracker = SortTracker("configs/config.yaml")
        
        detections = [{
            "bbox": [10, 10, 50, 50],
            "keypoints": {"nose": (30, 30, 0.9)},
            "track_id": None,
        }]
        
        result, _ = tracker.update(detections)
        assert len(result) == 1
        assert result[0]["track_id"] is not None

    def test_multiple_detections_create_multiple_tracks(self):
        """Multiple detections should create separate tracks."""
        Track.reset_counter()
        tracker = SortTracker("configs/config.yaml")
        tracker.min_hits = 1  # Assign IDs when hits >= 1
        
        detections = [
            {"bbox": [10, 10, 50, 50], "keypoints": {}, "track_id": None},
            {"bbox": [60, 60, 100, 100], "keypoints": {}, "track_id": None},
        ]
        
        result, _ = tracker.update(detections)
        assert len(result) == 2
        # Both should have valid IDs after first frame since min_hits=1
        # A track created with hits=1 should satisfy hits >= min_hits=1
        assert result[0]["track_id"] > 0, f"Expected ID > 0, got {result[0]['track_id']}"
        assert result[1]["track_id"] > 0, f"Expected ID > 0, got {result[1]['track_id']}"
        # Different track IDs
        assert result[0]["track_id"] != result[1]["track_id"]


class TestTrackMatching:
    """Test IoU-based track matching across frames."""

    def test_same_detection_keeps_id(self):
        """Detection in same location should keep track ID."""
        Track.reset_counter()
        tracker = SortTracker("configs/config.yaml")
        
        det1 = [{"bbox": [10, 10, 50, 50], "keypoints": {}, "track_id": None}]
        result1, _ = tracker.update(det1)
        track_id_1 = result1[0]["track_id"]
        
        # Frame 2: same detection (high IoU)
        det2 = [{"bbox": [11, 11, 51, 51], "keypoints": {}, "track_id": None}]
        result2, _ = tracker.update(det2)
        track_id_2 = result2[0]["track_id"]
        
        assert track_id_1 == track_id_2

    def test_moved_detection_keeps_id_if_iou_high(self):
        """Detection moved slightly should maintain ID."""
        Track.reset_counter()
        tracker = SortTracker("configs/config.yaml")
        
        det1 = [{"bbox": [10, 10, 50, 50], "keypoints": {}, "track_id": None}]
        result1, _ = tracker.update(det1)
        id1 = result1[0]["track_id"]
        
        # Moved 5 pixels right
        det2 = [{"bbox": [15, 10, 55, 50], "keypoints": {}, "track_id": None}]
        result2, _ = tracker.update(det2)
        id2 = result2[0]["track_id"]
        
        assert id1 == id2

    def test_far_detection_creates_new_id(self):
        """Detection far from previous should create new track."""
        Track.reset_counter()
        tracker = SortTracker("configs/config.yaml")
        tracker.min_hits = 1  # Immediate ID assignment
        
        det1 = [{"bbox": [10, 10, 50, 50], "keypoints": {}, "track_id": None}]
        result1, _ = tracker.update(det1)
        id1 = result1[0]["track_id"]
        
        # Far away → low IoU
        det2 = [{"bbox": [200, 200, 240, 240], "keypoints": {}, "track_id": None}]
        result2, _ = tracker.update(det2)
        id2 = result2[0]["track_id"]
        
        assert id1 > 0
        assert id2 > 0
        assert id1 != id2


class TestTrackExpiration:
    """Test track removal after max_age frames without match."""

    def test_track_expires_after_max_age(self):
        """Track should be removed after max_age frames with no detection."""
        Track.reset_counter()
        tracker = SortTracker("configs/config.yaml")
        tracker.max_age = 2  # Override for test
        
        det1 = [{"bbox": [10, 10, 50, 50], "keypoints": {}, "track_id": None}]
        tracker.update(det1)
        
        # 3 empty frames → should expire
        for _ in range(3):
            tracker.update([])
        
        assert len(tracker.tracks) == 0

    def test_expired_ids_returned(self):
        """Expired track IDs should be returned for cleanup."""
        Track.reset_counter()
        tracker = SortTracker("configs/config.yaml")
        tracker.max_age = 1
        tracker.min_hits = 1
        
        det1 = [{"bbox": [10, 10, 50, 50], "keypoints": {}, "track_id": None}]
        result1, _ = tracker.update(det1)
        id1 = result1[0]["track_id"]
        
        # Empty frame
        result2, expired = tracker.update([])
        # Track should still exist (frames_since_update = 1)
        assert id1 not in expired
        
        # Another empty frame → now expired
        result3, expired2 = tracker.update([])
        assert id1 in expired2
