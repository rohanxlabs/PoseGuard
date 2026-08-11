"""
Test temporal smoothing, velocity calculation, and debouncing.
"""
import pytest
from collections import deque
from src.temporal import TemporalAnalyzer


class TestTemporalSmoothing:
    """Test moving average smoothing of keypoints."""

    def test_smoothing_single_frame(self):
        """Single frame returns the same keypoints."""
        analyzer = TemporalAnalyzer(window=3)
        kps = {"nose": (100, 50, 0.9), "left_eye": (90, 45, 0.85)}
        
        analyzer.update(track_id=1, keypoints=kps)
        smoothed = analyzer.get_smoothed_keypoints(1)
        
        assert smoothed is not None
        assert smoothed["nose"][0] == 100
        assert smoothed["nose"][1] == 50

    def test_smoothing_averages_over_window(self):
        """Multiple frames should average coordinates."""
        analyzer = TemporalAnalyzer(window=3)
        track_id = 1
        
        # Frame 1: nose at (100, 50)
        analyzer.update(track_id, {"nose": (100, 50, 0.9)})
        # Frame 2: nose at (102, 50)
        analyzer.update(track_id, {"nose": (102, 50, 0.9)})
        # Frame 3: nose at (104, 50)
        analyzer.update(track_id, {"nose": (104, 50, 0.9)})
        
        smoothed = analyzer.get_smoothed_keypoints(track_id)
        # Average should be (102, 50)
        assert smoothed is not None
        assert 101 < smoothed["nose"][0] < 103

    def test_velocity_calculation(self):
        """Velocity should reflect coordinate change."""
        analyzer = TemporalAnalyzer(window=3)
        track_id = 1
        
        analyzer.update(track_id, {"nose": (100, 50, 0.9)})
        analyzer.update(track_id, {"nose": (110, 50, 0.9)})
        
        velocities = analyzer.get_velocities(track_id)
        assert velocities is not None
        # Nose moved 10 pixels horizontally
        assert velocities["nose"] > 9

    def test_nonexistent_track_returns_none(self):
        """Querying unknown track should return None or empty dict."""
        analyzer = TemporalAnalyzer(window=3)
        assert analyzer.get_smoothed_keypoints(999) is None
        # Velocities returns empty dict for unknown tracks
        vels = analyzer.get_velocities(999)
        assert vels is None or vels == {}


class TestEventDebouncing:
    """Test event debouncing logic."""

    def test_first_event_emitted(self):
        """First occurrence of an event should be emitted."""
        analyzer = TemporalAnalyzer(window=3, debounce_frames=30)
        assert analyzer.should_emit_event(1, "knee_risk", frame_idx=100) is True

    def test_duplicate_suppressed_within_window(self):
        """Same event within debounce window should be suppressed."""
        analyzer = TemporalAnalyzer(window=3, debounce_frames=30)
        
        analyzer.should_emit_event(1, "knee_risk", frame_idx=100)
        # Same event 10 frames later → suppressed
        assert analyzer.should_emit_event(1, "knee_risk", frame_idx=110) is False

    def test_duplicate_emitted_after_window(self):
        """Same event after debounce window should be emitted."""
        analyzer = TemporalAnalyzer(window=3, debounce_frames=30)
        
        analyzer.should_emit_event(1, "knee_risk", frame_idx=100)
        # 31 frames later → emitted again
        assert analyzer.should_emit_event(1, "knee_risk", frame_idx=131) is True

    def test_different_injury_types_independent(self):
        """Different injury types should have independent debounce."""
        analyzer = TemporalAnalyzer(window=3, debounce_frames=30)
        
        analyzer.should_emit_event(1, "knee_risk", frame_idx=100)
        # Different injury type in same window → emitted
        assert analyzer.should_emit_event(1, "elbow_risk", frame_idx=105) is True

    def test_different_tracks_independent(self):
        """Different tracks should have independent debounce."""
        analyzer = TemporalAnalyzer(window=3, debounce_frames=30)
        
        analyzer.should_emit_event(1, "knee_risk", frame_idx=100)
        # Same injury type, different track → emitted
        assert analyzer.should_emit_event(2, "knee_risk", frame_idx=105) is True


class TestTrackReset:
    """Test track state cleanup when a person leaves frame."""

    def test_reset_clears_history(self):
        """Resetting a track should clear temporal history."""
        analyzer = TemporalAnalyzer(window=3)
        track_id = 1
        
        analyzer.update(track_id, {"nose": (100, 50, 0.9)})
        assert analyzer.get_smoothed_keypoints(track_id) is not None
        
        analyzer.reset_track(track_id)
        assert analyzer.get_smoothed_keypoints(track_id) is None

    def test_reset_clears_debounce_state(self):
        """Resetting should clear debounce history for that track."""
        analyzer = TemporalAnalyzer(window=3, debounce_frames=30)
        
        analyzer.should_emit_event(1, "knee_risk", frame_idx=100)
        assert analyzer.should_emit_event(1, "knee_risk", frame_idx=110) is False
        
        # Reset track
        analyzer.reset_track(1)
        # Same event should now be emitted (debounce cleared)
        assert analyzer.should_emit_event(1, "knee_risk", frame_idx=115) is True
