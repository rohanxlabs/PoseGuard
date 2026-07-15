import cv2
import time
import yaml
import logging

from src.detector import PoseDetector
from src.pose_analyzer import PoseAnalyzer
from src.injury_classifier import InjuryClassifier
from src.tracker import SortTracker
from src.temporal import TemporalAnalyzer
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """
    Real-time inference on webcam or RTSP stream.
    Press 'q' to quit, 's' to save current frame.
    """
    def __init__(self, config_path: str = "configs/config.yaml", source: int = 0):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.source = source

        self.detector   = PoseDetector(config_path)
        self.analyzer   = PoseAnalyzer(config_path)
        self.classifier = InjuryClassifier(config_path)
        self.tracker    = SortTracker(config_path)
        self.temporal   = TemporalAnalyzer(window=self.cfg["pose"]["smoothing_window"])
        self.visualizer = Visualizer(config_path)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(f"Starting realtime pipeline on source: {self.source}")

        frame_idx = 0
        fps = 30.0
        t_prev = time.time()
        total_events = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            persons = self.detector.detect(frame)
            persons = self.tracker.update(persons)

            for person in persons:
                tid = person.get("track_id", -1)
                kps = person["keypoints"]
                self.temporal.update(tid, kps)
                smoothed = self.temporal.get_smoothed_keypoints(tid) or kps
                velocities = self.temporal.get_velocities(tid)
                angles    = self.analyzer.compute_angles(smoothed)
                asymmetry = self.analyzer.limb_asymmetry(smoothed)
                events    = self.classifier.classify(angles, asymmetry, smoothed, velocities, tid, frame_idx)

                risk = max((e.risk_score for e in events), default=0)
                _, color = self.classifier.get_risk_level(risk)

                frame = self.visualizer.draw_skeleton(frame, person, color)
                frame = self.visualizer.draw_bbox(frame, person, color, tid)
                frame = self.visualizer.draw_injury_alerts(frame, events, person)
                frame = self.visualizer.draw_angles(frame, person, angles)
                total_events += len([e for e in events if self.temporal.should_emit_event(tid, e.injury_type, frame_idx)])

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            frame = self.visualizer.draw_hud(frame, frame_idx, total_events, fps)

            cv2.imshow("Injury Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"outputs/frame_{saved_count:04d}.jpg", frame)
                saved_count += 1

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()