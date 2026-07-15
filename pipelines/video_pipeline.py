import cv2
import time
import json
import logging
import yaml
import os
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from src.detector import PoseDetector
from src.pose_analyzer import PoseAnalyzer
from src.injury_classifier import InjuryClassifier
from src.tracker import SortTracker
from src.temporal import TemporalAnalyzer
from src.visualizer import Visualizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VideoInjuryPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg_path = config_path
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        logger.info("Initializing pipeline components...")
        self.detector   = PoseDetector(config_path)
        self.analyzer   = PoseAnalyzer(config_path)
        self.classifier = InjuryClassifier(config_path)
        self.tracker    = SortTracker(config_path)
        self.temporal   = TemporalAnalyzer(
            window=self.cfg["pose"]["smoothing_window"]
        )
        self.visualizer = Visualizer(config_path)

        self.event_log: List[dict] = []
        self.total_events = 0

    def run(self, input_path: str, output_path: Optional[str] = None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path and self.cfg["output"]["save_video"]:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

        logger.info(f"Processing: {input_path} | {width}x{height} @ {fps_in:.1f}fps | {total} frames")

        frame_idx = 0
        fps_display = fps_in
        t_prev = time.time()

        with tqdm(total=total, desc="Analyzing", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- Detect ---
                persons = self.detector.detect(frame)

                # --- Track ---
                persons = self.tracker.update(persons)

                for person in persons:
                    tid = person.get("track_id", -1)
                    kps = person["keypoints"]

                    # --- Temporal update ---
                    self.temporal.update(tid, kps)
                    smoothed_kps = self.temporal.get_smoothed_keypoints(tid) or kps
                    velocities   = self.temporal.get_velocities(tid)

                    # --- Analyze ---
                    angles    = self.analyzer.compute_angles(smoothed_kps)
                    asymmetry = self.analyzer.limb_asymmetry(smoothed_kps)

                    # --- Classify ---
                    raw_events = self.classifier.classify(
                        angles, asymmetry, smoothed_kps,
                        velocities, tid, frame_idx
                    )

                    # Debounce events
                    events = [
                        e for e in raw_events
                        if self.temporal.should_emit_event(tid, e.injury_type, frame_idx)
                    ]

                    if events:
                        self.total_events += len(events)
                        for e in events:
                            self.event_log.append({
                                "frame": frame_idx,
                                "track_id": tid,
                                "type": e.injury_type,
                                "label": e.label,
                                "score": round(e.risk_score, 3),
                                "details": e.details,
                            })
                            logger.warning(str(e))

                    # --- Visualize ---
                    risk_score = max((e.risk_score for e in raw_events), default=0)
                    level, color = self.classifier.get_risk_level(risk_score)

                    frame = self.visualizer.draw_skeleton(frame, person, color)
                    frame = self.visualizer.draw_bbox(frame, person, color, tid)
                    frame = self.visualizer.draw_injury_alerts(frame, raw_events, person)
                    frame = self.visualizer.draw_angles(frame, person, angles)

                # FPS calculation
                now = time.time()
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now

                frame = self.visualizer.draw_hud(frame, frame_idx, self.total_events, fps_display)

                if writer:
                    writer.write(frame)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        if writer:
            writer.release()

        self._save_report(output_path)
        logger.info(f"Done. Total injury events: {self.total_events}")
        return self.event_log

    def _save_report(self, output_path: Optional[str]):
        if not self.cfg["output"]["save_logs"] or not self.event_log:
            return
        report_path = Path("outputs/logs/injury_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(self.event_log, f, indent=2)
        logger.info(f"Report saved: {report_path}")