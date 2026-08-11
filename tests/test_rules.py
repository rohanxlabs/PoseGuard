"""
Test injury risk classification rules.
Mock the classifier to avoid requiring full pose detection.
"""
import pytest
from src.injury_classifier import InjuryClassifier, INJURY_TYPES


class TestRiskLevels:
    """Test risk level assignment based on score."""

    def test_high_risk_threshold(self):
        """Score >= 0.75 should be HIGH risk."""
        classifier = InjuryClassifier("configs/config.yaml")
        level, color = classifier.get_risk_level(0.80)
        assert level == "HIGH"
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_medium_risk_threshold(self):
        """Score >= 0.45 and < 0.75 should be MEDIUM risk."""
        classifier = InjuryClassifier("configs/config.yaml")
        level, color = classifier.get_risk_level(0.60)
        assert level == "MEDIUM"
        assert isinstance(color, tuple)

    def test_low_risk_threshold(self):
        """Score < 0.45 should be LOW risk."""
        classifier = InjuryClassifier("configs/config.yaml")
        level, color = classifier.get_risk_level(0.30)
        assert level == "LOW"
        assert isinstance(color, tuple)

    def test_boundary_high(self):
        """Score exactly at HIGH threshold."""
        classifier = InjuryClassifier("configs/config.yaml")
        level, _ = classifier.get_risk_level(0.75)
        assert level == "HIGH"

    def test_boundary_medium(self):
        """Score exactly at MEDIUM threshold."""
        classifier = InjuryClassifier("configs/config.yaml")
        level, _ = classifier.get_risk_level(0.45)
        assert level == "MEDIUM"


class TestInjuryTypes:
    """Test injury type definitions."""

    def test_injury_types_defined(self):
        """All expected injury types should exist."""
        expected = [
            "NONE", "ACL_RISK", "ANKLE_SPRAIN", "HAMSTRING",
            "SHOULDER", "ELBOW", "SPINE",
            "FALL", "ASYMMETRY"
        ]
        for injury in expected:
            assert injury in INJURY_TYPES

    def test_injury_labels_human_readable(self):
        """Each injury type should have a readable label."""
        for injury_type, label in INJURY_TYPES.items():
            assert isinstance(label, str)
            assert len(label) > 0


class TestKneeHyperextension:
    """Test knee hyperextension rule."""

    def test_knee_hyperextension_detected(self):
        """Knee angle > threshold should trigger ACL risk."""
        classifier = InjuryClassifier("configs/config.yaml")
        angles = {
            "left_knee": 175,  # Hyperextended
            "right_knee": 150,
        }
        events = classifier.classify(
            angles=angles,
            asymmetry={},
            keypoints={},
            velocities={},
            track_id=1,
            frame_idx=100
        )
        
        # Should detect left knee risk
        acl_events = [e for e in events if e.injury_type == "ACL_RISK"]
        assert len(acl_events) > 0
        assert "left" in acl_events[0].details.lower() or "knee" in acl_events[0].details.lower()

    def test_normal_knee_angle_no_event(self):
        """Normal knee angle should not trigger event."""
        classifier = InjuryClassifier("configs/config.yaml")
        angles = {
            "left_knee": 160,
            "right_knee": 155,
        }
        events = classifier.classify(
            angles=angles,
            asymmetry={},
            keypoints={},
            velocities={},
            track_id=1,
            frame_idx=100
        )
        
        acl_events = [e for e in events if e.injury_type == "ACL_RISK"]
        assert len(acl_events) == 0


class TestAsymmetryDetection:
    """Test limb asymmetry rule."""

    def test_asymmetry_detected(self):
        """High asymmetry ratio should trigger event."""
        classifier = InjuryClassifier("configs/config.yaml")
        asymmetry = {
            "left_thigh": 100.0,
            "right_thigh": 70.0,
            "thigh_ratio": 0.30,  # 30% asymmetry
        }
        events = classifier.classify(
            angles={},
            asymmetry=asymmetry,
            keypoints={},
            velocities={},
            track_id=1,
            frame_idx=100
        )
        
        asym_events = [e for e in events if e.injury_type == "ASYMMETRY"]
        assert len(asym_events) > 0

    def test_balanced_limbs_no_event(self):
        """Balanced limbs should not trigger asymmetry."""
        classifier = InjuryClassifier("configs/config.yaml")
        # Don't pass individual limbs — only pass ratio keys
        asymmetry = {
            "thigh_ratio": 0.02,  # 2% asymmetry — below threshold
            "shin_ratio": 0.01,
            "arm_ratio": 0.03,
        }
        events = classifier.classify(
            angles={},
            asymmetry=asymmetry,
            keypoints={},
            velocities={},
            track_id=1,
            frame_idx=100
        )
        
        asym_events = [e for e in events if e.injury_type == "ASYMMETRY"]
        assert len(asym_events) == 0


class TestVelocitySpike:
    """Test velocity spike detection."""

    def test_high_velocity_detected(self):
        """High joint velocity should trigger event."""
        classifier = InjuryClassifier("configs/config.yaml")
        velocities = {
            "left_knee": 95.0,  # High velocity
            "right_knee": 10.0,
        }
        # Classifier needs 3+ frames of velocity history for spike detection
        # Prime the history
        classifier._update_velocity_history(1, {"left_knee": 10.0})
        classifier._update_velocity_history(1, {"left_knee": 15.0})
        classifier._update_velocity_history(1, velocities)
        
        events = classifier.classify(
            angles={},
            asymmetry={},
            keypoints={},
            velocities=velocities,
            track_id=1,
            frame_idx=100
        )
        
        fall_events = [e for e in events if e.injury_type == "FALL"]
        assert len(fall_events) > 0

    def test_normal_velocity_no_event(self):
        """Normal velocities should not trigger spike."""
        classifier = InjuryClassifier("configs/config.yaml")
        velocities = {
            "left_knee": 30.0,
            "right_knee": 25.0,
        }
        events = classifier.classify(
            angles={},
            asymmetry={},
            keypoints={},
            velocities=velocities,
            track_id=1,
            frame_idx=100
        )
        
        fall_events = [e for e in events if e.injury_type == "FALL"]
        assert len(fall_events) == 0
