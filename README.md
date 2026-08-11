# 🛡️ PoseGuard

**Real-time biomechanical risk signal detection from sports video using human pose estimation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚠️ Important Disclaimer

**PoseGuard is a computer vision research and prototyping system — NOT a medical device.**

Risk events detected by PoseGuard are heuristic signals derived from configurable biomechanical angle and velocity thresholds. These signals:

- Have **not** been clinically validated
- Do **not** constitute medical diagnoses
- Do **not** predict injuries with scientifically proven accuracy
- Should **not** be used for clinical decision-making

**Always consult a qualified sports medicine professional for injury assessment and prevention.**

---

## 📋 Overview

PoseGuard is a **computer vision pipeline** that combines:
- **YOLOv8-Pose** for human pose estimation
- **Multi-person tracking** using IoU-based SORT-style matching
- **Temporal motion analysis** with smoothing and velocity calculation
- **Biomechanical feature extraction** (joint angles, limb asymmetry)
- **Configurable rule-based risk classifier** for movement pattern analysis

The system processes sports video to identify movement patterns that may correlate with biomechanical stress signals commonly associated with injury risk in sports medicine literature.

---

## 🎯 Problem Statement

Traditional sports injury risk assessment requires:
- Manual video review by trained professionals
- Frame-by-frame biomechanical analysis
- Subjective interpretation of movement patterns
- Post-hoc analysis after injury occurs

PoseGuard demonstrates how computer vision can automate the detection of specific biomechanical signals (knee hyperextension, asymmetry, sudden falls) for research and exploratory analysis in sports movement science.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────┐
│   Video Input / Webcam / RTSP Stream   │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│          YOLOv8-Pose Detector          │
│     17 COCO Keypoints per Person       │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│       IoU-Based Person Tracker         │
│       Assigns Stable Track IDs         │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│        Temporal Analyzer               │
│   Moving Average Smoothing + Velocity  │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│      Biomechanics Feature Engine       │
│  Joint Angles + Limb Asymmetry + Fall  │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│     Rule-Based Risk Classifier         │
│    8 Configurable Threshold Rules      │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│        Event Debouncing + Log          │
└──────────────┬─────────────────────────┘
               ↓
┌────────────────────────────────────────┐
│   Visualization + JSON/CSV Reports     │
└────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Implementation | Status |
|---------|----------------|--------|
| **YOLOv8-Pose Detection** | Ultralytics YOLO with 17-keypoint COCO format | ✅ Verified |
| **Multi-Person Detection** | Processes all detected persons per frame | ✅ Verified |
| **Person Tracking** | IoU-based SORT-style tracker with Hungarian assignment | ✅ Verified |
| **Temporal Smoothing** | Moving-average filter over configurable window | ✅ Verified |
| **Velocity Calculation** | Frame-to-frame keypoint displacement | ✅ Verified |
| **Joint Angle Analysis** | Knee, hip, elbow, shoulder, spine angles | ✅ Verified |
| **Ankle Inversion Proxy** | Shin-to-vertical tilt angle | ✅ Verified |
| **Limb Asymmetry** | Left/right thigh, shin, arm length comparison | ✅ Verified |
| **Fall Detection** | Rapid torso drop over N-frame window | ✅ Verified |
| **Velocity Spike Detection** | Joint acceleration threshold crossing | ✅ Verified |
| **Event Debouncing** | Suppresses duplicate events within time window | ✅ Verified |
| **Configurable Thresholds** | All rules adjustable via `config.yaml` | ✅ Verified |
| **Risk Scoring** | HIGH (≥0.75) / MEDIUM (≥0.45) / LOW (<0.45) | ✅ Verified |
| **Video Pipeline** | Batch processing with progress bar | ✅ Verified |
| **Real-Time Pipeline** | Webcam/RTSP with live overlay | ✅ Verified |
| **Streamlit Dashboard** | Upload video, configure thresholds, view timeline | ✅ Verified |
| **JSON Reports** | Structured event log with frame/track/score | ✅ Verified |
| **CSV Reports** | Tabular export for analysis | ✅ Verified |
| **Test Suite** | 45 unit tests (geometry, temporal, tracking, rules) | ✅ Verified |

---

## 🚨 Risk Signal Rules

PoseGuard implements **8 configurable biomechanical threshold rules**:

| Rule | Signal | Default Threshold | Event Type | Biomechanical Basis |
|------|--------|-------------------|------------|---------------------|
| **1. Knee Hyperextension** | Knee angle | > 170° | ACL / Ligament Risk | Hyperextension increases ACL strain |
| **2. Elbow Hyperextension** | Elbow angle | > 175° | Elbow Injury Risk | Valgus stress on medial collateral ligament |
| **3. Hip Abduction** | Hip abduction angle | > 50° | Hamstring Strain Risk | Excessive abduction during running/cutting |
| **4. Spine Lateral Flexion** | Trunk lean angle | > 40° | Spine/Back Stress | Lateral overload on lumbar spine |
| **5. Ankle Inversion** | Shin tilt from vertical | > 35° | Ankle Sprain Risk | Inversion moment at ankle joint |
| **6. Velocity Spike** | Joint velocity | > 80 px/frame | Fall / Impact | Sudden acceleration indicates fall/collision |
| **7. Torso Drop** | Vertical torso displacement | > 120 px over 6 frames | Fall Detected | Rapid descent of center of mass |
| **8. Limb Asymmetry** | Left/right limb length ratio | > 0.25 | Compensatory Pattern | Imbalance suggests guarding or weakness |

**Note:** Default thresholds are heuristic estimates. They should be tuned for specific camera angles, athlete populations, and sport contexts through empirical validation.

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- (Optional) CUDA-compatible GPU for real-time performance

### Setup

```bash
# Clone repository
git clone https://github.com/rohanxlabs/PoseGuard.git
cd PoseGuard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# YOLO weights will download automatically on first run
```

---

## 🚀 Usage

### 1. Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

**Workflow:**
1. Configure model (YOLOv8n/s/m/l) and thresholds in sidebar
2. Click **"⚡ Build Pipeline"**
3. Upload a sports video (MP4, AVI, MOV, MKV)
4. Click **"▶ Run Analysis"**
5. View live processing, risk timeline, and events
6. Download JSON/CSV reports from **Report** tab

**Real-Time Mode:**
- Switch to **"🔴 Realtime"** tab
- Enter camera source (0 for webcam, or RTSP URL)
- Click **"▶ Start Feed"**

---

### 2. Command-Line Interface

#### Video Processing
```bash
python main.py \
  --mode video \
  --input data/raw/video.mp4 \
  --output outputs/videos/analyzed.mp4
```

#### Real-Time Webcam
```bash
python main.py --mode realtime --source 0
```

#### RTSP Stream
```bash
python main.py --mode realtime --source rtsp://192.168.1.100:554/stream
```

---

### 3. Configuration

Edit `configs/config.yaml` to adjust:

```yaml
model:
  weights: "yolov8m-pose.pt"  # n/s/m/l/x
  confidence: 0.5
  device: "cuda"  # or "cpu"

injury:
  knee_hyperextension_angle: 170
  ankle_inversion_angle: 35
  asymmetry_threshold: 0.25
  velocity_spike_threshold: 80
  # ... (see config.yaml for full list)
```

---

## 📊 Output Reports

### JSON Format (`outputs/logs/injury_report.json`)
```json
[
  {
    "frame": 142,
    "track_id": 2,
    "type": "ACL_RISK",
    "label": "ACL / Knee Ligament Risk",
    "risk_level": "HIGH",
    "score": 0.82,
    "details": "Left knee: 176.3°"
  }
]
```

### CSV Format (`outputs/logs/injury_report.csv`)
```
frame,track_id,type,label,risk_level,score,details
142,2,ACL_RISK,ACL / Knee Ligament Risk,HIGH,0.82,Left knee: 176.3°
```

---

## 🧪 Testing

Run the test suite (no YOLO weights required):

```bash
pytest tests/ -v
```

**Test Coverage:**
- ✅ Geometric calculations (angles, distances)
- ✅ Temporal smoothing and velocity
- ✅ Tracker ID stability and expiration
- ✅ Event debouncing
- ✅ Risk classification rules

---

## 📸 Annotation Tool

Extract frames with keypoint overlays for manual labeling:

```bash
python scripts/annotate_tool.py \
  --input data/raw/video.mp4 \
  --output data/annotations/frames/ \
  --skip 10
```

**Controls:**
- `s` — Save current frame + JSON metadata
- `n` — Next frame
- `q` — Quit

---

## ⚙️ Performance

Performance depends on:
- Model size (yolov8n is fastest, yolov8x is most accurate)
- Hardware (GPU strongly recommended for real-time)
- Input resolution
- Number of people in frame

**Approximate FPS (single person, 640×480, YOLOv8m):**
- **CPU (Intel i7):** ~3-5 FPS
- **GPU (RTX 3060):** ~25-30 FPS

**Recommendation:** Use YOLOv8n or YOLOv8s on CPU; YOLOv8m or YOLOv8l on GPU.

---

## ⚠️ Limitations

1. **Heuristic Thresholds:** Risk rules are based on biomechanics literature but have not been validated on clinical outcome data. Sensitivity and specificity are unknown.

2. **2D Projection:** Camera captures 2D projections of 3D movements. Out-of-plane rotations (e.g., internal tibial rotation) are not detectable.

3. **Camera Dependency:** Accuracy depends on:
   - Camera angle (sagittal/frontal plane alignment)
   - Lighting conditions
   - Distance from subject
   - Lens distortion

4. **Pose Estimation Errors:** YOLOv8-Pose can produce:
   - False negatives (missed keypoints in occlusion)
   - False positives (incorrect keypoint localization)
   - Low confidence on partial visibility

5. **Ankle Angle Proxy:** True ankle inversion requires 3D joint angles. PoseGuard uses shin-to-vertical tilt as a 2D proxy.

6. **No Ground Truth:** Without motion capture or clinical follow-up, the system cannot be validated for predictive accuracy.

7. **Tracking Fragility:** IoU-based tracking can:
   - Lose IDs during occlusion
   - Swap IDs when people cross paths
   - Fail on crowded scenes

8. **Single-Camera:** Multi-view analysis would improve 3D angle estimation but is not implemented.

9. **Real-Time Latency:** CPU inference may have 200-500ms latency, unsuitable for immediate intervention.

10. **MLP Classifier:** The optional `models/injury_mlp.py` is **reference code only** — it is not trained or used during inference.

---

## 🛠️ Technology Stack

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — Pose estimation
- **[OpenCV](https://opencv.org/)** — Video I/O and visualization
- **[NumPy](https://numpy.org/)** — Numerical computation
- **[SciPy](https://scipy.org/)** — Hungarian algorithm for tracking
- **[PyTorch](https://pytorch.org/)** — Deep learning backend
- **[Streamlit](https://streamlit.io/)** — Interactive dashboard
- **[Plotly](https://plotly.com/)** — Risk timeline charts
- **[pytest](https://pytest.org/)** — Test framework

---

## 📁 Project Structure

```
PoseGuard/
├── app.py                      # Streamlit dashboard
├── main.py                     # CLI entry point
├── configs/
│   └── config.yaml             # Thresholds and model settings
├── src/
│   ├── detector.py             # YOLOv8-Pose wrapper
│   ├── pose_analyzer.py        # Joint angle calculations
│   ├── tracker.py              # Multi-person IoU tracker
│   ├── temporal.py             # Smoothing, velocity, debouncing
│   ├── injury_classifier.py    # Rule-based risk engine
│   └── visualizer.py           # Skeleton + bbox + HUD rendering
├── pipelines/
│   ├── video_pipeline.py       # Batch video processing
│   └── realtime_pipeline.py    # Webcam/RTSP processing
├── models/
│   └── injury_mlp.py           # (Optional) PyTorch MLP (unused)
├── scripts/
│   ├── train_classifier.py     # MLP training script (reference)
│   └── annotate_tool.py        # Frame annotation helper
├── tests/
│   ├── test_geometry.py        # Angle calculation tests
│   ├── test_temporal.py        # Smoothing/velocity tests
│   ├── test_tracker.py         # Tracking tests
│   └── test_rules.py           # Risk classifier tests
├── data/
│   ├── raw/                    # Input videos (not included)
│   └── annotations/            # Labeled frames (user-generated)
├── outputs/
│   ├── videos/                 # Processed videos
│   └── logs/                   # JSON/CSV reports
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🎓 Use Cases

✅ **Research & Education:**
- Demonstrating pose estimation pipelines
- Teaching biomechanics feature extraction
- Prototyping sports analytics systems

✅ **Sports Video Analysis:**
- Exploratory movement pattern detection
- Comparative analysis across athletes
- Generating timestamped event logs for coach review

❌ **NOT Suitable For:**
- Medical diagnosis or clinical decision-making
- Real-time injury prevention systems without validation
- Automated athlete selection/exclusion
- Insurance or liability decisions

---

## 🚧 Future Work

Potential extensions (not currently implemented):

- [ ] Multi-camera 3D reconstruction for true joint angles
- [ ] Temporal Convolutional Networks (TCN) for sequence classification
- [ ] Integration with wearable IMU data for validation
- [ ] Sport-specific rule profiles (basketball, soccer, athletics)
- [ ] Kalman filter tracking for smoother ID stability
- [ ] Attention visualization (grad-CAM on pose heatmaps)
- [ ] Cloud deployment with batch processing API
- [ ] Clinical validation study with ground-truth injury outcomes

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

**Commercial Use:** Permitted under MIT license terms, but **clinical or medical use requires regulatory approval** (e.g., FDA 510(k) in the USA, CE marking in EU) and is **not recommended** without proper validation studies.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request with clear description

**Code Style:** Follow PEP 8. Run `pytest` before committing.

---

## 📧 Contact

**Author:** Rohan  
**Repository:** [github.com/rohanxlabs/PoseGuard](https://github.com/rohanxlabs/PoseGuard)

For questions, issues, or collaboration inquiries, please open a GitHub issue.

---

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv8-Pose
- COCO dataset for keypoint annotation standard
- SORT tracking algorithm by Alex Bewley
- Sports medicine research community for biomechanics insights

---

## 📚 References

1. Bewley, A., et al. "Simple Online and Realtime Tracking." *ICIP 2016*.
2. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." *CVPR 2016*.
3. Lin, T., et al. "Microsoft COCO: Common Objects in Context." *ECCV 2014*.
4. Ultralytics YOLOv8 Documentation: [docs.ultralytics.com](https://docs.ultralytics.com/)

**Note:** PoseGuard is inspired by biomechanics principles from sports medicine literature, but the specific threshold values and risk classifications have not been peer-reviewed or clinically validated.

---

**⚡ Built with YOLOv8-Pose · Engineered for Research · Not for Clinical Use**
