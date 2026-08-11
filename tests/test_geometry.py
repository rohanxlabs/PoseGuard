"""
Test geometric calculations: angles, distances, asymmetry.
These tests verify the core biomechanics math without requiring YOLO.
"""
import pytest
import numpy as np
from src.pose_analyzer import angle_between, shin_from_vertical_angle


class TestAngleBetween:
    """Test joint angle calculations."""

    def test_right_angle(self):
        """90-degree angle at vertex."""
        p1 = (0, 0, 1.0)
        vertex = (1, 0, 1.0)
        p2 = (1, 1, 1.0)
        angle = angle_between(p1, vertex, p2)
        assert angle is not None
        assert 89 < angle < 91

    def test_straight_line(self):
        """180-degree angle (straight line)."""
        p1 = (0, 0, 1.0)
        vertex = (1, 0, 1.0)
        p2 = (2, 0, 1.0)
        angle = angle_between(p1, vertex, p2)
        assert angle is not None
        assert 179 < angle < 181

    def test_acute_angle(self):
        """45-degree acute angle (correctly constructed)."""
        # vertex at origin, p1 along +x axis, p2 at 45deg
        vertex = (0, 0, 1.0)
        p1 = (1, 0, 1.0)
        p2 = (np.cos(np.radians(45)), np.sin(np.radians(45)), 1.0)
        angle = angle_between(p1, vertex, p2)
        assert angle is not None
        assert 44 < angle < 46

    def test_none_point_returns_none(self):
        """Missing keypoint should return None."""
        assert angle_between(None, (1, 0, 1.0), (2, 0, 1.0)) is None
        assert angle_between((0, 0, 1.0), None, (2, 0, 1.0)) is None
        assert angle_between((0, 0, 1.0), (1, 0, 1.0), None) is None

    def test_zero_length_vector(self):
        """Coincident points should return None."""
        p1 = (0, 0, 1.0)
        vertex = (0, 0, 1.0)
        p2 = (1, 1, 1.0)
        assert angle_between(p1, vertex, p2) is None


class TestShinAngle:
    """Test ankle inversion proxy (shin-to-vertical angle)."""

    def test_vertical_shin(self):
        """Perfectly vertical shin → 0 degrees."""
        knee  = (100, 50, 1.0)
        ankle = (100, 100, 1.0)  # directly below
        angle = shin_from_vertical_angle(knee, ankle)
        assert angle is not None
        assert angle < 1  # near 0

    def test_tilted_shin(self):
        """Shin tilted 30 degrees from vertical."""
        knee  = (100, 50, 1.0)
        # ankle displaced horizontally by tan(30°) * 50
        ankle = (100 + 50 * np.tan(np.radians(30)), 100, 1.0)
        angle = shin_from_vertical_angle(knee, ankle)
        assert angle is not None
        assert 29 < angle < 31

    def test_horizontal_shin(self):
        """Shin pointing horizontally → 90 degrees."""
        knee  = (100, 100, 1.0)
        ankle = (150, 100, 1.0)
        angle = shin_from_vertical_angle(knee, ankle)
        assert angle is not None
        assert 89 < angle < 91

    def test_none_keypoint(self):
        """Missing keypoint should return None."""
        assert shin_from_vertical_angle(None, (100, 100, 1.0)) is None
        assert shin_from_vertical_angle((100, 50, 1.0), None) is None

    def test_coincident_points(self):
        """Knee == ankle should return None."""
        assert shin_from_vertical_angle((100, 100, 1.0), (100, 100, 1.0)) is None
