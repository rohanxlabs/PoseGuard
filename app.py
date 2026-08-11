"""
PoseGuard — Biomechanical Risk Signal Detection
Streamlit dashboard.  Run: streamlit run app.py

Architecture: this UI delegates ALL pose, tracking, temporal, biomechanics,
and risk logic to the src/ modules — no inline reimplementation.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import json
import yaml
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional

import plotly.graph_objects as go
import pandas as pd

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PoseGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #050810; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #070c1a !important; border-right: 1px solid rgba(0,210,255,0.10); }
.pg-hero { background: linear-gradient(135deg,#07111f 0%,#0b1d35 60%,#07111f 100%);
  border: 1px solid rgba(0,210,255,0.18); border-radius: 18px;
  padding: 38px 48px 30px; margin-bottom: 28px; position: relative; overflow: hidden; }
.pg-logo { font-family:'Syne',sans-serif; font-weight:800; font-size:2.8rem;
  letter-spacing:-1.5px; background:linear-gradient(90deg,#00d2ff 0%,#0077ff 60%,#00ffcc 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1; }
.pg-tagline { font-family:'JetBrains Mono',monospace; font-size:0.78rem;
  color:rgba(0,210,255,0.6); letter-spacing:3px; text-transform:uppercase; margin-top:6px; }
.pg-badge { display:inline-block; background:rgba(0,210,255,0.08);
  border:1px solid rgba(0,210,255,0.25); color:#00d2ff;
  font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:1.5px;
  padding:4px 12px; border-radius:100px; margin-top:14px; margin-right:8px; }
.disclaimer-box { background:rgba(255,176,32,0.07); border:1px solid rgba(255,176,32,0.35);
  border-radius:12px; padding:16px 20px; margin-bottom:20px;
  font-family:'JetBrains Mono',monospace; font-size:0.70rem; color:rgba(255,176,32,0.9);
  line-height:1.7; }
.metric-card { background:#070c1a; border:1px solid rgba(255,255,255,0.07);
  border-radius:14px; padding:20px 22px; position:relative; overflow:hidden; }
.metric-label { font-family:'JetBrains Mono',monospace; font-size:0.65rem;
  letter-spacing:2px; color:rgba(255,255,255,0.35); text-transform:uppercase; margin-bottom:6px; }
.metric-value { font-family:'Syne',sans-serif; font-weight:700; font-size:2rem; line-height:1; }
.metric-value.cyan{color:#00d2ff;} .metric-value.red{color:#ff3c50;}
.metric-value.amber{color:#ffb020;} .metric-value.green{color:#00e87a;}
.section-header { font-family:'Syne',sans-serif; font-weight:700; font-size:0.78rem;
  letter-spacing:3px; text-transform:uppercase; color:rgba(0,210,255,0.6);
  margin:28px 0 14px; }
.stButton>button { background:linear-gradient(135deg,#0077ff,#00d2ff) !important;
  color:#050810 !important; font-family:'Syne',sans-serif !important; font-weight:700 !important;
  border:none !important; border-radius:10px !important; padding:10px 28px !important; }
div[data-testid="stFileUploadDropzone"] { background:#070c1a !important;
  border-color:rgba(0,210,255,0.22) !important; border-radius:14px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG_PATH = "configs/config.yaml"

@st.cache_data
def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

cfg = _load_config()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "event_log":             [],
        "frame_count":           0,
        "total_events":          0,
        "high_count":            0,
        "medium_count":          0,
        "low_count":             0,
        "risk_timeline":         [],
        "person_count_history":  [],
        "pipeline":              None,   # lazily built PipelineBundle
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _reset_counters():
    for k in ["event_log", "risk_timeline", "person_count_history"]:
        st.session_state[k] = []
    for k in ["frame_count", "total_events", "high_count", "medium_count", "low_count"]:
        st.session_state[k] = 0


def get_risk_color_hex(level: str) -> str:
    return {"HIGH": "#ff3c50", "MEDIUM": "#ffb020", "LOW": "#00e87a"}.get(level, "#aaa")

# ─────────────────────────────────────────────
# PIPELINE BUNDLE  (built once per session)
# ─────────────────────────────────────────────
class PipelineBundle:
    """
    Wraps all src/ components so the Streamlit app uses the exact same
    detector → tracker → temporal → analyser → classifier chain as the
    CLI pipeline.  Shared state (tracker, temporal) is reset between runs.
    """
    def __init__(self, weights: str, conf: float, kp_conf: float,
                 threshold_overrides: dict):
        # Patch config in memory so all components pick up UI values
        patched_cfg = _load_config()
        patched_cfg["model"]["weights"]    = weights
        patched_cfg["model"]["confidence"] = conf
        patched_cfg["pose"]["keypoint_confidence_threshold"] = kp_conf
        for k, v in threshold_overrides.items():
            patched_cfg["injury"][k] = v

        # Write patched config to a temp location components can read
        import tempfile, os
        self._tmp_cfg = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False)
        yaml.dump(patched_cfg, self._tmp_cfg)
        self._tmp_cfg.flush()
        tmp_path = self._tmp_cfg.name

        from src.detector          import PoseDetector
        from src.pose_analyzer     import PoseAnalyzer
        from src.tracker           import SortTracker, Track
        from src.temporal          import TemporalAnalyzer
        from src.injury_classifier import InjuryClassifier
        from src.visualizer        import Visualizer

        Track.reset_counter()
        self.detector   = PoseDetector(tmp_path)
        self.analyzer   = PoseAnalyzer(tmp_path)
        self.classifier = InjuryClassifier(tmp_path)
        self.tracker    = SortTracker(tmp_path)
        self.temporal   = TemporalAnalyzer(
            window=patched_cfg["pose"]["smoothing_window"])
        self.visualizer = Visualizer(tmp_path)

    def process_frame(self, frame: np.ndarray, frame_idx: int,
                      show_skeleton: bool, show_bbox: bool, show_angles: bool):
        """
        Run one frame through the full pipeline.
        Returns (annotated_frame, frame_events, person_count).
        """
        persons = self.detector.detect(frame)
        persons, expired_ids = self.tracker.update(persons)

        for eid in expired_ids:
            self.temporal.reset_track(eid)

        frame_events = []
        person_count = len(persons)

        for person in persons:
            tid = person.get("track_id", -1)
            kps = person["keypoints"]

            self.temporal.update(tid, kps)
            smoothed   = self.temporal.get_smoothed_keypoints(tid) or kps
            velocities = self.temporal.get_velocities(tid)

            angles    = self.analyzer.compute_angles(smoothed)
            asymmetry = self.analyzer.limb_asymmetry(smoothed)
            raw_events = self.classifier.classify(
                angles, asymmetry, smoothed, velocities, tid, frame_idx)

            debounced = [
                e for e in raw_events
                if self.temporal.should_emit_event(tid, e.injury_type, frame_idx)
            ]

            risk = max((e.risk_score for e in raw_events), default=0.0)
            level, color = self.classifier.get_risk_level(risk)

            if show_skeleton:
                frame = self.visualizer.draw_skeleton(frame, person, color)
            if show_bbox:
                frame = self.visualizer.draw_bbox(frame, person, color, tid)
            if debounced:
                frame = self.visualizer.draw_injury_alerts(frame, debounced, person)
            if show_angles:
                frame = self.visualizer.draw_angles(frame, person, angles)

            for e in debounced:
                lv, _ = self.classifier.get_risk_level(e.risk_score)
                frame_events.append({
                    "frame":      frame_idx,
                    "track_id":   tid,
                    "type":       e.injury_type,
                    "label":      e.label,
                    "risk_level": lv,
                    "score":      round(e.risk_score, 3),
                    "detail":     e.details,
                })

        return frame, frame_events, person_count

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 24px'>
      <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.5rem;
                  background:linear-gradient(90deg,#00d2ff,#0077ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  letter-spacing:-0.5px'>🛡️ PoseGuard</div>
      <div style='font-family:"JetBrains Mono",monospace;font-size:0.62rem;
                  color:rgba(0,210,255,0.45);letter-spacing:2.5px;
                  text-transform:uppercase;margin-top:4px'>Risk Signal Detection</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### ⚙️ Model")
    model_choice = st.selectbox(
        "YOLOv8 Weights",
        ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt"],
        index=2)
    conf_thresh = st.slider("Detection Confidence", 0.25, 0.90,
                            float(cfg["model"]["confidence"]), 0.05)
    kp_conf     = st.slider("Keypoint Confidence",  0.20, 0.80,
                            float(cfg["pose"]["keypoint_confidence_threshold"]), 0.05)

    st.markdown("---")
    st.markdown("### 🚨 Risk Thresholds")
    inj = cfg["injury"]
    t_knee   = st.slider("Knee Hyperextension (°)",  155, 185,
                         int(inj["knee_hyperextension_angle"]))
    t_elbow  = st.slider("Elbow Hyperextension (°)", 160, 185,
                         int(inj["elbow_hyperextension_angle"]))
    t_hip    = st.slider("Hip Abduction (°)",          30,  80,
                         int(inj["hip_abduction_angle"]))
    t_spine  = st.slider("Spine Lateral Flex (°)",     20,  60,
                         int(inj["spine_lateral_flexion_angle"]))
    t_ankle  = st.slider("Ankle Inversion (°)",         5,  45,
                         int(inj["ankle_inversion_angle"]))
    t_asym   = st.slider("Asymmetry Ratio",           0.10, 0.60,
                         float(inj["asymmetry_threshold"]), 0.05)

    threshold_overrides = {
        "knee_hyperextension_angle":   t_knee,
        "elbow_hyperextension_angle":  t_elbow,
        "hip_abduction_angle":         t_hip,
        "spine_lateral_flexion_angle": t_spine,
        "ankle_inversion_angle":       t_ankle,
        "asymmetry_threshold":         t_asym,
    }

    st.markdown("---")
    st.markdown("### 🎥 Output")
    show_skeleton = st.toggle("Draw Skeleton",     value=True)
    show_angles   = st.toggle("Show Joint Angles", value=True)
    show_bbox     = st.toggle("Show Bounding Box", value=True)
    save_report   = st.toggle("Save JSON Report",  value=True)
    process_every = st.slider("Process Every N Frames", 1, 6, 2,
                               help="Skip frames for speed")

    if st.button("⚡ Build Pipeline"):
        with st.spinner("Loading model & initialising pipeline..."):
            try:
                st.session_state.pipeline = PipelineBundle(
                    model_choice, conf_thresh, kp_conf, threshold_overrides)
                st.success(f"✅ {model_choice} ready")
            except Exception as exc:
                st.error(f"❌ Pipeline init failed: {exc}")
                st.session_state.pipeline = None

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="pg-hero">
  <div class="pg-logo">🛡️ PoseGuard</div>
  <div class="pg-tagline">Biomechanical Risk Signal Detection from Sports Video</div>
  <div style="margin-top:16px">
    <span class="pg-badge">YOLOv8-Pose</span>
    <span class="pg-badge">Biomechanics Engine</span>
    <span class="pg-badge">Multi-Person Tracking</span>
    <span class="pg-badge">8-Rule Risk Classifier</span>
  </div>
</div>
<div class="disclaimer-box">
  ⚠️ <strong>Research Tool — Not a Medical Device.</strong>
  Risk events are heuristic signals derived from configurable angle and velocity thresholds.
  They do not constitute clinical diagnosis of injury and should not be used as such.
  Always consult a qualified sports-medicine professional for injury assessment.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_analyze, tab_realtime, tab_report, tab_about = st.tabs([
    "📹  Analyze Video", "🔴  Realtime", "📊  Report", "ℹ️  About",
])

# ════════════════════════════════════════════════
# TAB 1 — ANALYZE VIDEO
# ════════════════════════════════════════════════
with tab_analyze:
    col1, col2, col3, col4 = st.columns(4)
    def _metric(col, label, val, cls):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value {cls}">{val}</div></div>',
            unsafe_allow_html=True)
    _metric(col1, "Frames Processed", f'{st.session_state.frame_count:,}',  "cyan")
    _metric(col2, "Risk Events",       st.session_state.total_events,        "red")
    _metric(col3, "High Risk",         st.session_state.high_count,          "amber")
    _metric(col4, "Medium Risk",       st.session_state.medium_count,        "green")

    st.markdown("<div class='section-header'>Upload Video</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop your sports video here",
                                type=["mp4","avi","mov","mkv"],
                                label_visibility="collapsed")

    pipeline: Optional[PipelineBundle] = st.session_state.pipeline
    if pipeline is None:
        st.info("💡 Click **⚡ Build Pipeline** in the sidebar before running analysis.")

    if uploaded and pipeline is not None:
        col_run, col_reset = st.columns([1, 5])
        with col_run:
            run_btn = st.button("▶  Run Analysis", use_container_width=True)
        with col_reset:
            if st.button("↺  Reset"):
                _reset_counters()
                st.rerun()

        if run_btn:
            _reset_counters()

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.close()

            cap          = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.markdown("<div class='section-header'>Live Preview</div>",
                        unsafe_allow_html=True)
            c_vid, c_events = st.columns([3, 2])
            with c_vid:
                frame_display = st.empty()
                prog_bar      = st.progress(0)
                status_txt    = st.empty()
            with c_events:
                st.markdown(
                    '<div style=\'font-family:"JetBrains Mono",monospace;font-size:0.68rem;'
                    'color:rgba(0,210,255,0.55);letter-spacing:2px;'
                    'text-transform:uppercase;margin-bottom:12px\'>Live Events</div>',
                    unsafe_allow_html=True)
                event_container = st.empty()

            frame_idx    = 0
            event_log    = []
            t_start      = time.time()
            recent_evts: deque = deque(maxlen=8)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % process_every != 0:
                    continue

                st.session_state.frame_count += 1
                elapsed  = time.time() - t_start
                proc_fps = st.session_state.frame_count / max(elapsed, 1e-3)

                frame, frame_events, person_count = pipeline.process_frame(
                    frame, frame_idx, show_skeleton, show_bbox, show_angles)

                # HUD
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (320,75), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                cv2.putText(frame, "POSEGUARD", (8,18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,210,255), 2, cv2.LINE_AA)
                cv2.putText(frame,
                            f"Frame:{frame_idx}  FPS:{proc_fps:.1f}  Persons:{person_count}",
                            (8,38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,180,180), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Events:{st.session_state.total_events}", (8,58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (50,60,255) if st.session_state.total_events else (0,200,100),
                            1, cv2.LINE_AA)

                for ev in frame_events:
                    st.session_state.total_events += 1
                    if ev["risk_level"] == "HIGH":
                        st.session_state.high_count   += 1
                    elif ev["risk_level"] == "MEDIUM":
                        st.session_state.medium_count += 1
                    else:
                        st.session_state.low_count    += 1
                    event_log.append(ev)
                    recent_evts.appendleft(ev)

                st.session_state.risk_timeline.append(
                    (frame_idx, max((e["score"] for e in frame_events), default=0.0)))
                st.session_state.person_count_history.append((frame_idx, person_count))

                frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    channels="RGB", use_container_width=True)
                prog_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
                status_txt.markdown(
                    f'<div style="display:inline-flex;align-items:center;gap:7px;'
                    f'background:rgba(0,232,122,0.10);border:1px solid rgba(0,232,122,0.28);'
                    f'color:#00e87a;font-family:JetBrains Mono,monospace;font-size:0.68rem;'
                    f'letter-spacing:1.5px;padding:5px 14px;border-radius:100px">'
                    f'<div style="width:6px;height:6px;background:#00e87a;border-radius:50%"></div>'
                    f'Processing · Frame {frame_idx}/{total_frames} · {proc_fps:.1f} FPS</div>',
                    unsafe_allow_html=True)

                if recent_evts:
                    html = ""
                    for ev in list(recent_evts)[:6]:
                        c = get_risk_color_hex(ev["risk_level"])
                        html += (f'<div style="background:#07111f;border-left:3px solid {c};'
                                 f'border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:6px">'
                                 f'<div style="font-weight:600;font-size:0.78rem;color:#f0f4ff">'
                                 f'{ev["label"]}</div>'
                                 f'<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;'
                                 f'color:rgba(255,255,255,0.35);margin-top:2px">'
                                 f'Frame {ev["frame"]} · Track {ev["track_id"]} · '
                                 f'{ev["risk_level"]} · {ev["score"]:.0%}</div></div>')
                    event_container.markdown(html, unsafe_allow_html=True)

            cap.release()
            st.session_state.event_log = event_log
            status_txt.markdown(
                '<div style="display:inline-flex;background:rgba(0,210,255,0.08);'
                'border:1px solid rgba(0,210,255,0.3);color:#00d2ff;'
                'font-family:JetBrains Mono,monospace;font-size:0.68rem;'
                'padding:5px 14px;border-radius:100px">✅ Analysis Complete</div>',
                unsafe_allow_html=True)

            if save_report and event_log:
                Path("outputs/logs").mkdir(parents=True, exist_ok=True)
                with open("outputs/logs/poseguard_report.json", "w") as f:
                    json.dump(event_log, f, indent=2)

            # Risk timeline chart
            if st.session_state.risk_timeline:
                st.markdown("<div class='section-header'>Risk Score Timeline</div>",
                            unsafe_allow_html=True)
                frames_t = [x[0] for x in st.session_state.risk_timeline]
                scores_t = [x[1] for x in st.session_state.risk_timeline]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=frames_t, y=scores_t, mode="lines",
                    line=dict(color="#00d2ff", width=1.5), fill="tozeroy",
                    fillcolor="rgba(0,210,255,0.07)", name="Risk Score"))
                fig.add_hline(y=0.75, line_dash="dash",
                              line_color="rgba(255,60,80,0.5)",
                              annotation_text="HIGH", annotation_font_color="#ff3c50")
                fig.add_hline(y=0.45, line_dash="dash",
                              line_color="rgba(255,176,32,0.5)",
                              annotation_text="MEDIUM", annotation_font_color="#ffb020")
                fig.update_layout(
                    paper_bgcolor="#050810", plot_bgcolor="#070c1a",
                    font=dict(family="JetBrains Mono", color="#aab4c8", size=11),
                    margin=dict(l=40,r=20,t=20,b=40), height=220,
                    xaxis=dict(title="Frame", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.05)",
                               range=[0,1]),
                    showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    elif uploaded and pipeline is None:
        st.warning("⚠️ Build the pipeline from the sidebar first.")
    else:
        st.markdown("""
        <div style='background:#070c1a;border:1.5px dashed rgba(0,210,255,0.22);
                    border-radius:16px;padding:48px 32px;text-align:center'>
          <div style='font-size:3rem;margin-bottom:12px'>🎬</div>
          <div style='font-weight:700;font-size:1.1rem;color:#e2e8f0;margin-bottom:6px'>
            Drop a sports video to begin</div>
          <div style='font-family:"JetBrains Mono",monospace;font-size:0.7rem;
                      color:rgba(255,255,255,0.3);letter-spacing:1px'>
            MP4 · AVI · MOV · MKV</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 2 — REALTIME
# ════════════════════════════════════════════════
with tab_realtime:
    st.markdown("""
    <div style='background:#070c1a;border:1px solid rgba(0,210,255,0.12);
                border-radius:14px;padding:28px 32px;margin-bottom:20px'>
      <div style='font-size:1.1rem;font-weight:700;color:#f0f4ff;margin-bottom:8px'>
        🔴 Real-time Camera Feed</div>
      <div style='font-family:"JetBrains Mono",monospace;font-size:0.72rem;
                  color:rgba(255,255,255,0.35);line-height:1.8'>
        Supports webcam (0) and RTSP streams.<br>
        Build the pipeline from the sidebar, then press Start.<br>
        Press <b style="color:#f0f4ff">Stop</b> or close the tab to end the session.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer-box">⚠️ Risk events shown here are heuristic signals '
        'only — not clinical diagnoses.</div>', unsafe_allow_html=True)

    cam_source = st.text_input("Camera Source", value="0",
                               help="0 = default webcam, or paste RTSP URL")
    rt_col1, rt_col2 = st.columns(2)
    with rt_col1:
        start_rt = st.button("▶  Start Feed", use_container_width=True)
    with rt_col2:
        stop_rt  = st.button("⏹  Stop",       use_container_width=True)

    if start_rt:
        rt_pipeline: Optional[PipelineBundle] = st.session_state.pipeline
        if rt_pipeline is None:
            st.error("Build the pipeline from the sidebar first.")
        else:
            source = int(cam_source) if cam_source.strip().isdigit() else cam_source
            cap    = cv2.VideoCapture(source)
            if not cap.isOpened():
                st.error(f"Cannot open source: {cam_source}")
            else:
                rt_frame  = st.empty()
                rt_status = st.empty()
                rt_events = st.empty()
                recent_rt: deque = deque(maxlen=6)
                frame_idx = 0
                t_start   = time.time()

                while not stop_rt:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if frame_idx % process_every != 0:
                        continue

                    frame, frame_events, _ = rt_pipeline.process_frame(
                        frame, frame_idx, show_skeleton, show_bbox, show_angles)

                    fps_rt = frame_idx / max(time.time() - t_start, 1e-3)
                    cv2.putText(frame, f"PoseGuard LIVE | {fps_rt:.1f} fps",
                                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,210,255), 2)

                    rt_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   channels="RGB", use_container_width=True)
                    rt_status.markdown(
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;'
                        f'color:#00e87a">● LIVE · {fps_rt:.1f} FPS</div>',
                        unsafe_allow_html=True)

                    for ev in frame_events:
                        recent_rt.appendleft(ev)
                    if recent_rt:
                        html = "".join(
                            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;'
                            f'color:{get_risk_color_hex(e["risk_level"])};padding:3px 0">'
                            f'⚠ {e["label"]} · {e["score"]:.0%}</div>'
                            for e in list(recent_rt)[:4])
                        rt_events.markdown(html, unsafe_allow_html=True)

                cap.release()

# ════════════════════════════════════════════════
# TAB 3 — REPORT
# ════════════════════════════════════════════════
with tab_report:
    st.markdown("<div class='section-header'>Analysis Report</div>",
                unsafe_allow_html=True)
    events = st.session_state.event_log

    if not events:
        st.markdown("""
        <div style='background:#070c1a;border:1px solid rgba(255,255,255,0.06);
                    border-radius:14px;padding:40px;text-align:center;
                    color:rgba(255,255,255,0.25);font-family:"JetBrains Mono",monospace;
                    font-size:0.78rem;letter-spacing:1.5px'>
          NO DATA YET — RUN AN ANALYSIS FIRST
        </div>""", unsafe_allow_html=True)
    else:
        df     = pd.DataFrame(events)
        counts = df["risk_level"].value_counts()

        r1, r2, r3, r4 = st.columns(4)
        _metric(r1, "Total Events", len(df),                  "cyan")
        _metric(r2, "High Risk",    counts.get("HIGH",   0),  "red")
        _metric(r3, "Medium Risk",  counts.get("MEDIUM", 0),  "amber")
        _metric(r4, "Low Risk",     counts.get("LOW",    0),  "green")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        c_pie, c_bar = st.columns(2)

        with c_pie:
            st.markdown("<div class='section-header'>Signal Type Distribution</div>",
                        unsafe_allow_html=True)
            tc = df["label"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=tc.index, values=tc.values, hole=0.55,
                marker=dict(colors=["#00d2ff","#ff3c50","#ffb020","#00e87a",
                                     "#a855f7","#f97316","#ec4899","#3b82f6"]),
                textfont=dict(family="JetBrains Mono", size=11)))
            fig_pie.update_layout(
                paper_bgcolor="#050810", plot_bgcolor="#050810",
                font=dict(family="JetBrains Mono", color="#aab4c8"),
                margin=dict(l=10,r=10,t=10,b=10), height=260,
                legend=dict(font=dict(size=10)), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_bar:
            st.markdown("<div class='section-header'>Events by Risk Level</div>",
                        unsafe_allow_html=True)
            risk_order = ["HIGH","MEDIUM","LOW"]
            risk_vals  = [counts.get(r, 0) for r in risk_order]
            fig_bar = go.Figure(go.Bar(
                x=risk_order, y=risk_vals,
                marker_color=["#ff3c50","#ffb020","#00e87a"],
                text=risk_vals, textposition="auto",
                textfont=dict(family="JetBrains Mono", size=13, color="#050810")))
            fig_bar.update_layout(
                paper_bgcolor="#050810", plot_bgcolor="#070c1a",
                font=dict(family="JetBrains Mono", color="#aab4c8"),
                margin=dict(l=20,r=20,t=10,b=20), height=260,
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<div class='section-header'>Full Event Log</div>",
                    unsafe_allow_html=True)

        def _style_risk(val):
            colors = {"HIGH": "#ff3c50", "MEDIUM": "#ffb020", "LOW": "#00e87a"}
            c = colors.get(val, "#aaa")
            return f"color:{c};font-weight:600;font-family:JetBrains Mono,monospace"

        display_cols = ["frame","track_id","label","risk_level","score","detail"]
        rename_map   = {"frame":"Frame","track_id":"Track","label":"Signal Type",
                        "risk_level":"Risk","score":"Score","detail":"Detail"}
        styled = (df[display_cols].rename(columns=rename_map)
                  .style
                  .map(_style_risk, subset=["Risk"])
                  .format({"Score": "{:.0%}"}))
        st.dataframe(styled, use_container_width=True, height=320)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇  Download JSON",
                               data=json.dumps(events, indent=2),
                               file_name="poseguard_report.json",
                               mime="application/json",
                               use_container_width=True)
        with dl2:
            st.download_button("⬇  Download CSV",
                               data=df.to_csv(index=False),
                               file_name="poseguard_report.csv",
                               mime="text/csv",
                               use_container_width=True)

# ════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<div style='max-width:820px'>

<div style='font-size:1.6rem;font-weight:800;color:#f0f4ff;
            letter-spacing:-0.5px;margin-bottom:6px'>
  PoseGuard — Biomechanical Risk Signal Detection</div>
<div style='font-family:"JetBrains Mono",monospace;font-size:0.72rem;
            color:rgba(0,210,255,0.55);letter-spacing:2px;margin-bottom:28px'>
  YOLOV8-POSE · BIOMECHANICS ENGINE · RULE-BASED RISK CLASSIFIER
</div>

<div style='background:rgba(255,176,32,0.07);border:1px solid rgba(255,176,32,0.35);
            border-radius:12px;padding:20px 24px;margin-bottom:20px;
            font-family:"JetBrains Mono",monospace;font-size:0.73rem;
            color:rgba(255,176,32,0.9);line-height:1.8'>
  ⚠️ <strong>Important Disclaimer</strong><br>
  PoseGuard is a computer-vision research and prototyping system, not a medical device.<br>
  Risk events are heuristic signals derived from configurable biomechanical angle and
  velocity thresholds. They have not been clinically validated and must not be interpreted
  as diagnoses or predictions of injury. Consult a qualified sports-medicine professional
  for any injury-related decision.
</div>

<div style='background:#070c1a;border:1px solid rgba(0,210,255,0.10);
            border-radius:14px;padding:28px 32px;margin-bottom:18px'>
  <div style='font-weight:700;font-size:0.9rem;color:#00d2ff;margin-bottom:12px'>
    🔬 Pipeline Architecture</div>
  <div style='font-family:"JetBrains Mono",monospace;font-size:0.75rem;
              color:rgba(255,255,255,0.55);line-height:2.2'>
    Video / Webcam<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    YOLOv8-Pose &nbsp;→ 17 COCO keypoints per person<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    IoU Tracker &nbsp;&nbsp;→ stable track IDs across frames<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Temporal Engine → moving-average smoothing + velocity<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Biomechanics &nbsp;→ joint angles + limb asymmetry<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Risk Rules &nbsp;&nbsp;&nbsp;→ 8 configurable threshold checks<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Event Debounce → suppress duplicate events<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Visualisation + JSON / CSV Report
  </div>
</div>

<div style='background:#070c1a;border:1px solid rgba(255,60,80,0.12);
            border-radius:14px;padding:28px 32px;margin-bottom:18px'>
  <div style='font-weight:700;font-size:0.9rem;color:#ff3c50;margin-bottom:14px'>
    🚨 Risk Signal Rules (all thresholds configurable)</div>
  <table style='width:100%;border-collapse:collapse;font-size:0.78rem;
                font-family:"JetBrains Mono",monospace'>
    <tr style='color:rgba(0,210,255,0.6);border-bottom:1px solid rgba(255,255,255,0.06)'>
      <th style='text-align:left;padding:8px 0'>Signal</th>
      <th style='text-align:left;padding:8px 0'>Default Threshold</th>
      <th style='text-align:left;padding:8px 0'>Event Type</th>
    </tr>
    <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
      <td style='padding:8px 0'>Knee angle</td><td style='color:#ffb020'>&gt; 170°</td>
      <td style='color:#ff3c50'>ACL / Ligament Risk</td></tr>
    <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
      <td style='padding:8px 0'>Elbow angle</td><td style='color:#ffb020'>&gt; 175°</td>
      <td style='color:#ff3c50'>Elbow Hyperextension</td></tr>
    <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
      <td style='padding:8px 0'>Hip abduction</td><td style='color:#ffb020'>&gt; 50°</td>
      <td style='color:#ff3c50'>Hamstring Strain</td></tr>
    <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
      <td style='padding:8px 0'>Spine lateral flex</td><td style='color:#ffb020'>&gt; 40°</td>
      <td style='color:#ff3c50'>Spine / Back Stress</td></tr>
    <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
      <td style='padding:8px 0'>Ankle inversion (shin tilt)</td><td style='color:#ffb020'>&gt; 35°</td>
      <td style='color:#ff3c50'>Ankle Sprain Risk</td></tr>
    <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
      <td style='padding:8px 0'>Joint velocity spike</td><td style='color:#ffb020'>&gt; 80 px/frame</td>
      <td style='color:#ff3c50'>Fall / Impact</td></tr>
    <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
      <td style='padding:8px 0'>Torso drop (6 frames)</td><td style='color:#ffb020'>&gt; 120 px</td>
      <td style='color:#ff3c50'>Fall Detected</td></tr>
    <tr style='color:rgba(255,255,255,0.6)'>
      <td style='padding:8px 0'>Limb asymmetry ratio</td><td style='color:#ffb020'>&gt; 0.25</td>
      <td style='color:#ff3c50'>Compensatory Asymmetry</td></tr>
  </table>
</div>

<div style='background:#070c1a;border:1px solid rgba(255,255,255,0.07);
            border-radius:14px;padding:24px 32px;margin-bottom:18px'>
  <div style='font-weight:700;font-size:0.9rem;color:#f0f4ff;margin-bottom:12px'>
    ⚠️ Known Limitations</div>
  <ul style='font-size:0.80rem;color:rgba(255,255,255,0.55);line-height:2;margin:0;
             padding-left:18px;font-family:"JetBrains Mono",monospace'>
    <li>Thresholds are heuristic — not derived from a validated clinical dataset.</li>
    <li>Pose estimation accuracy degrades with occlusion, camera angle, and poor lighting.</li>
    <li>2D projections cannot capture out-of-plane joint mechanics.</li>
    <li>Ankle inversion uses a shin-tilt proxy, not a true ankle-joint angle.</li>
    <li>CPU inference is slow; GPU recommended for near-real-time performance.</li>
    <li>The optional MLP classifier is reference code only — not active during inference.</li>
  </ul>
</div>

<div style='background:#070c1a;border:1px solid rgba(255,255,255,0.07);
            border-radius:14px;padding:24px 32px'>
  <div style='font-weight:700;font-size:0.9rem;color:#f0f4ff;margin-bottom:12px'>
    🛠 Tech Stack</div>
  <div style='display:flex;gap:10px;flex-wrap:wrap'>""" +
    "".join(
        f'<span style="background:rgba(0,210,255,0.08);border:1px solid rgba(0,210,255,0.2);'
        f'color:#00d2ff;font-family:JetBrains Mono,monospace;font-size:0.68rem;'
        f'padding:5px 14px;border-radius:100px">{t}</span>'
        for t in ["YOLOv8-Pose","Ultralytics","PyTorch","OpenCV",
                  "NumPy","Streamlit","Plotly","SciPy","PyYAML","pandas"]
    ) + """
  </div>
</div>

</div>""", unsafe_allow_html=True)
