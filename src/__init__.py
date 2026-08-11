"""
PoseGuard — src package
Public API for the core pipeline components.

Note: Imports are deferred to avoid forcing dependencies during test collection.
Import components directly when needed:
    from src.detector import PoseDetector
    from src.pose_analyzer import PoseAnalyzer
etc.
"""
__all__ = [
    "PoseDetector", "KEYPOINTS", "SKELETON_PAIRS",
    "PoseAnalyzer", "angle_between", "shin_from_vertical_angle",
    "SortTracker", "Track",
    "TemporalAnalyzer",
    "InjuryClassifier", "InjuryEvent", "INJURY_TYPES",
    "Visualizer",
]
