# 🛡️ PoseGuard — Real-time Athlete Injury Detection

PoseGuard analyzes sports video (live or recorded) and flags biomechanical risk
signals that precede common athletic injuries. It uses **YOLOv8-Pose** to detect
17 COCO keypoints per athlete, computes joint angles, smooths them over time,
and fires rule-based injury events (ACL, ankle sprain, hamstring, spine stress,
elbow hyperextension, falls, and limb asymmetry) with HIGH / MEDIUM / LOW risk
scoring.

---

## ✨ Features

- **Multi-person pose estimation** via YOLOv8-Pose (COCO 17-point skeleton).
- **Biomechanics engine** — real-time knee, hip, elbow, shoulder and spine angles.
- **Temporal smoothing** — moving-average keypoint denoising + fall/velocity tracking.
- **Rule-based risk classifier** — 9 injury types with configurable thresholds.
- **Multi-person tracking** — consistent athlete IDs across frames (SORT-style IoU tracker).
- **Streamlit dashboard** — upload a clip or use a webcam/RTSP feed, with live
  skeleton overlay, risk timeline, and downloadable JSON/CSV reports.
- **CLI pipeline** — headless batch analysis of a video file, or a live webcam window.

---

## 🗂️ Project Structure

```
PoseGuard/
├── app.py                      # Streamlit web UI (analyze / realtime / report / about)
├── main.py                     # CLI entry point (--mode video|realtime)
├── configs/
│   └── config.yaml             # model, pose, tracking, injury thresholds, output
├── pipelines/
│   ├── video_pipeline.py       # VideoInjuryPipeline (offline file analysis)
│   └── realtime_pipeline.py    # RealtimePipeline (webcam / RTSP)
├── src/
│   ├── detector.py             # YOLOv8-Pose wrapper
│   ├── pose_analyzer.py        # joint angles + limb asymmetry
│   ├── tracker.py              # IoU-based person tracker
│   ├── temporal.py             # smoothing, velocity, fall detection
│   ├── injury_classifier.py    # rule engine -> InjuryEvent objects
│   └── visualizer.py           # skeleton / bbox / HUD overlays
├── models/
│   └── injury_mlp.py           # optional learned classifier (reference)
├── scripts/
│   ├── annotate_tool.py        # annotation helper
│   └── train_classifier.py     # train the MLP classifier
├── data/
│   ├── raw/                    # input videos
│   └── annotations/            # ground-truth labels
└── outputs/
    ├── videos/                 # annotated output videos
    ├── logs/                   # injury_report.json
    └── frame_*.jpg             # saved realtime frames
```

---

## ⚙️ Installation

Requires **Python 3.10+** and a virtual environment (a `venv/` is already included).

```bash
# inside the project root
python -m venv venv
.\venv\Scripts\activate        # Windows  (source venv/bin/activate on Linux/macOS)

pip install -r requirements.txt
```

> Model weights (`yolov8m-pose.pt` by default) are downloaded automatically by
> Ultralytics the first time the model is loaded — an internet connection is
> required on first run.

> `config.yaml` ships with `model.device: "cpu"`. If you have a CUDA GPU and the
> CUDA build of PyTorch installed, set it to `"cuda"` for faster inference.

---

## 🚀 Usage

### Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

Then open **http://localhost:8501**:

1. In the sidebar pick the YOLOv8-Pose weights, confidence sliders and injury
   thresholds, then click **⚡ Load Model**.
2. **📹 Analyze Video** — upload an `.mp4/.avi/.mov/.mkv` clip and click
   **▶ Run Analysis**. A risk timeline and live event list appear.
3. **🔴 Realtime** — enter `0` for the default webcam (or an RTSP URL) and press
   **▶ Start Feed** (`q` quits, `s` saves a frame in the CLI version).
4. **📊 Report** — review type distribution, risk breakdown, full event log, and
   download JSON/CSV.

### Command Line

```bash
# Analyze a video file (writes annotated video + outputs/logs/injury_report.json)
python main.py --mode video --input data/raw/your_clip.mp4 --output outputs/videos/result.mp4

# Live webcam (0 = default camera, or pass an RTSP URL)
python main.py --mode realtime --source 0

# Custom config
python main.py --mode video --config configs/config.yaml
```

---

## 🚨 Injury Detection Rules

All thresholds live in `configs/config.yaml` under `injury:`.

| Signal                | Default Threshold | Injury Type            |
|-----------------------|-------------------|------------------------|
| Knee angle            | > 170°            | ACL / Ligament Risk    |
| Elbow angle           | > 175°            | Elbow Hyperextension   |
| Hip abduction         | > 50°             | Hamstring Strain       |
| Spine lateral flex    | > 40°             | Spine / Back Stress    |
| Ankle inversion       | > 35°             | Ankle Sprain Risk      |
| Velocity spike        | > 80 px/frame     | Fall / Impact          |
| Torso drop (6 frames) | > 120 px          | Fall Detected          |
| Limb asymmetry ratio  | > 0.25            | Compensatory Asymmetry |

Risk level is derived from the event score: **HIGH** ≥ 0.75, **MEDIUM** ≥ 0.45,
**LOW** otherwise.

---

## 🧩 How It Works

1. **Detection** — YOLOv8-Pose detects every person and returns 17 keypoints
   with confidence scores per frame.
2. **Tracking** — an IoU tracker assigns a stable `track_id` to each athlete.
3. **Temporal smoothing** — keypoint positions are averaged over a sliding window
   to suppress detection jitter; velocities and torso height are tracked for
   fall/impact detection.
4. **Biomechanics** — joint angles are computed via vector math on keypoint
   coordinates; left/right limb asymmetry is measured.
5. **Classification** — the rule engine compares angles/velocities against the
   configured thresholds and emits debounced `InjuryEvent` objects.
6. **Visualization** — skeletons, bounding boxes, joint-angle labels, and a HUD
   are drawn; reports are exported to JSON/CSV.

---

## 🛠️ Tech Stack

YOLOv8-Pose · Ultralytics · PyTorch · OpenCV · NumPy · Streamlit · Plotly ·
PyYAML · scikit-learn · pandas

---

## 📌 Notes

- The default `configs/config.yaml` expects `yolov8m-pose.pt`. Lighter/faster
  alternatives: `yolov8n-pose.pt`, `yolov8s-pose.pt`; heavier: `yolov8l-pose.pt`,
  `yolov8x-pose.pt`.
- For CPU-only setups, keep `model.device: "cpu"`; larger weights will be slow.
- `outputs/` and `data/raw/` start empty — provide your own footage for analysis.
