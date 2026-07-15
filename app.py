"""
PoseGuard — Real-time Athlete Injury Detection
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import json
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
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

.stApp {
    background: #050810;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #070c1a !important;
    border-right: 1px solid rgba(0,210,255,0.10);
}

/* ── Hero ── */
.pg-hero {
    background: linear-gradient(135deg, #07111f 0%, #0b1d35 60%, #07111f 100%);
    border: 1px solid rgba(0,210,255,0.18);
    border-radius: 18px;
    padding: 38px 48px 30px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.pg-hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -70px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(0,210,255,0.07) 0%, transparent 68%);
    pointer-events: none;
}
.pg-hero::after {
    content: '';
    position: absolute;
    bottom: -50px; left: 10px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(255,50,80,0.06) 0%, transparent 68%);
    pointer-events: none;
}
.pg-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg, #00d2ff 0%, #0077ff 60%, #00ffcc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.pg-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: rgba(0,210,255,0.6);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 6px;
}
.pg-badge {
    display: inline-block;
    background: rgba(0,210,255,0.08);
    border: 1px solid rgba(0,210,255,0.25);
    color: #00d2ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    padding: 4px 12px;
    border-radius: 100px;
    margin-top: 14px;
    margin-right: 8px;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.metric-card {
    background: #070c1a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(0,210,255,0.28); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 2px 2px 0 0;
}
.metric-card.cyan::before  { background: linear-gradient(90deg,#00d2ff,transparent); }
.metric-card.red::before   { background: linear-gradient(90deg,#ff3c50,transparent); }
.metric-card.amber::before { background: linear-gradient(90deg,#ffb020,transparent); }
.metric-card.green::before { background: linear-gradient(90deg,#00e87a,transparent); }

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    line-height: 1;
    color: #f0f4ff;
}
.metric-value.cyan  { color: #00d2ff; }
.metric-value.red   { color: #ff3c50; }
.metric-value.amber { color: #ffb020; }
.metric-value.green { color: #00e87a; }
.metric-sub {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.30);
    margin-top: 4px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(0,210,255,0.6);
    margin: 28px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0,210,255,0.10);
}

/* ── Event cards ── */
.event-card {
    background: #070c1a;
    border-left: 3px solid;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.event-card.HIGH   { border-color: #ff3c50; background: rgba(255,60,80,0.05); }
.event-card.MEDIUM { border-color: #ffb020; background: rgba(255,176,32,0.05); }
.event-card.LOW    { border-color: #00e87a; background: rgba(0,232,122,0.05); }

.event-type {
    font-weight: 600;
    font-size: 0.82rem;
    color: #f0f4ff;
}
.event-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: rgba(255,255,255,0.35);
    margin-top: 2px;
}
.risk-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 100px;
    letter-spacing: 1px;
}
.risk-pill.HIGH   { background: rgba(255,60,80,0.18);  color: #ff3c50; }
.risk-pill.MEDIUM { background: rgba(255,176,32,0.18); color: #ffb020; }
.risk-pill.LOW    { background: rgba(0,232,122,0.18);  color: #00e87a; }

/* ── Upload zone ── */
.upload-zone {
    background: #070c1a;
    border: 1.5px dashed rgba(0,210,255,0.22);
    border-radius: 16px;
    padding: 48px 32px;
    text-align: center;
}
.upload-icon {
    font-size: 3rem;
    margin-bottom: 12px;
    display: block;
}
.upload-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: #e2e8f0;
    margin-bottom: 6px;
}
.upload-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.3);
    letter-spacing: 1px;
}

/* ── Status pill ── */
.status-live {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(0,232,122,0.10);
    border: 1px solid rgba(0,232,122,0.28);
    color: #00e87a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    padding: 5px 14px;
    border-radius: 100px;
}
.status-dot {
    width: 6px; height: 6px;
    background: #00e87a;
    border-radius: 50%;
    animation: pulse 1.4s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.75); }
}

/* ── Table ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }
thead tr th {
    background: #070c1a !important;
    color: rgba(0,210,255,0.7) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── Streamlit overrides ── */
.stSlider > div > div { background: rgba(0,210,255,0.15) !important; }
.stSelectbox > div > div { background: #070c1a !important; border-color: rgba(0,210,255,0.18) !important; }
div[data-testid="stFileUploadDropzone"] {
    background: #070c1a !important;
    border-color: rgba(0,210,255,0.22) !important;
    border-radius: 14px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0077ff, #00d2ff) !important;
    color: #050810 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

div[data-testid="stProgress"] > div {
    background: rgba(0,210,255,0.12) !important;
    border-radius: 100px !important;
}
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0077ff, #00d2ff) !important;
    border-radius: 100px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KEYPOINTS + SKELETON (COCO 17-point)
# ─────────────────────────────────────────────
KEYPOINTS = {
    "nose":0,"left_eye":1,"right_eye":2,"left_ear":3,"right_ear":4,
    "left_shoulder":5,"right_shoulder":6,"left_elbow":7,"right_elbow":8,
    "left_wrist":9,"right_wrist":10,"left_hip":11,"right_hip":12,
    "left_knee":13,"right_knee":14,"left_ankle":15,"right_ankle":16,
}
SKELETON_PAIRS = [
    (5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
    (11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4),
]
INJURY_LABELS = {
    "ACL_RISK":     "ACL / Knee Ligament Risk",
    "ANKLE_SPRAIN": "Ankle Sprain Risk",
    "HAMSTRING":    "Hamstring Strain Risk",
    "FALL":         "Fall / Impact Detected",
    "SHOULDER":     "Shoulder Injury Risk",
    "SPINE":        "Spine / Back Stress",
    "ELBOW":        "Elbow Hyperextension",
    "ASYMMETRY":    "Compensatory Asymmetry",
}


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "event_log":      [],
        "frame_count":    0,
        "total_events":   0,
        "high_count":     0,
        "medium_count":   0,
        "low_count":      0,
        "processing":     False,
        "model_loaded":   False,
        "yolo_model":     None,
        "risk_timeline":  [],   # [(frame, score)]
        "person_count_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_model(weights: str):
    try:
        from ultralytics import YOLO
        model = YOLO(weights)
        st.session_state.yolo_model = model
        st.session_state.model_loaded = True
        return True
    except Exception as e:
        st.session_state.model_loaded = False
        return False


def angle_between(p1, vertex, p2) -> Optional[float]:
    if any(x is None for x in [p1, vertex, p2]):
        return None
    v1 = np.array([p1[0]-vertex[0], p1[1]-vertex[1]], dtype=np.float32)
    v2 = np.array([p2[0]-vertex[0], p2[1]-vertex[1]], dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1))))


def extract_keypoints(kp_raw, conf_thresh=0.4) -> Dict:
    kps = {}
    for name, idx in KEYPOINTS.items():
        x, y, c = kp_raw[idx]
        kps[name] = (float(x), float(y), float(c)) if c >= conf_thresh else None
    return kps


def compute_angles(kps: Dict) -> Dict:
    k = kps
    def pt(name): return k.get(name)
    return {
        "left_knee":    angle_between(pt("left_hip"),      pt("left_knee"),   pt("left_ankle")),
        "right_knee":   angle_between(pt("right_hip"),     pt("right_knee"),  pt("right_ankle")),
        "left_elbow":   angle_between(pt("left_shoulder"), pt("left_elbow"),  pt("left_wrist")),
        "right_elbow":  angle_between(pt("right_shoulder"),pt("right_elbow"), pt("right_wrist")),
        "left_hip":     angle_between(pt("left_shoulder"), pt("left_hip"),    pt("left_knee")),
        "right_hip":    angle_between(pt("right_shoulder"),pt("right_hip"),   pt("right_knee")),
        "spine_lateral": _spine_angle(kps),
    }


def _spine_angle(kps) -> Optional[float]:
    ls,rs = kps.get("left_shoulder"), kps.get("right_shoulder")
    lh,rh = kps.get("left_hip"),      kps.get("right_hip")
    if any(x is None for x in [ls,rs,lh,rh]):
        return None
    sm = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
    hm = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
    v  = np.array([sm[0]-hm[0], sm[1]-hm[1]], dtype=np.float32)
    vert = np.array([0,-1], dtype=np.float32)
    cos_a = np.clip(np.dot(v,vert)/(np.linalg.norm(v)+1e-6), -1, 1)
    return float(np.degrees(np.arccos(cos_a)))


def classify_injuries(angles, kps, thresholds, frame_idx, track_id=0) -> List[Dict]:
    events = []
    t = thresholds

    checks = [
        ("left_knee",   "ACL_RISK",     t["knee"],    "Left knee hyperextension"),
        ("right_knee",  "ACL_RISK",     t["knee"],    "Right knee hyperextension"),
        ("left_elbow",  "ELBOW",        t["elbow"],   "Left elbow hyperextension"),
        ("right_elbow", "ELBOW",        t["elbow"],   "Right elbow hyperextension"),
        ("left_hip",    "HAMSTRING",    t["hip"],     "Left hip abduction"),
        ("right_hip",   "HAMSTRING",    t["hip"],     "Right hip abduction"),
        ("spine_lateral","SPINE",       t["spine"],   "Spine lateral flexion"),
    ]

    for joint, inj_type, thresh, detail in checks:
        angle = angles.get(joint)
        if angle is not None and angle > thresh:
            excess = angle - thresh
            score = min(1.0, 0.5 + excess / 30)
            level = "HIGH" if score >= 0.75 else ("MEDIUM" if score >= 0.45 else "LOW")
            events.append({
                "frame":      frame_idx,
                "track_id":   track_id,
                "type":       inj_type,
                "label":      INJURY_LABELS[inj_type],
                "score":      round(score, 3),
                "risk_level": level,
                "detail":     f"{detail}: {angle:.1f}°",
            })

    return events


def draw_skeleton_on_frame(frame, kps, color=(0,230,120)):
    kp_list = [kps.get(name) for name in KEYPOINTS.keys()]
    for i, j in SKELETON_PAIRS:
        if kp_list[i] and kp_list[j]:
            cv2.line(frame,
                     (int(kp_list[i][0]), int(kp_list[i][1])),
                     (int(kp_list[j][0]), int(kp_list[j][1])),
                     color, 2, cv2.LINE_AA)
    for kp in kp_list:
        if kp:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255,255,255), -1)
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, 1)
    return frame


def get_risk_color_cv(level):
    return {"HIGH": (50,60,255), "MEDIUM": (32,176,255), "LOW": (122,232,0)}.get(level, (200,200,200))


def get_risk_color_hex(level):
    return {"HIGH": "#ff3c50", "MEDIUM": "#ffb020", "LOW": "#00e87a"}.get(level, "#aaa")


def risk_level(score):
    return "HIGH" if score >= 0.75 else ("MEDIUM" if score >= 0.45 else "LOW")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 24px'>
      <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.5rem;
                  background:linear-gradient(90deg,#00d2ff,#0077ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  letter-spacing:-0.5px'>
        🛡️ PoseGuard
      </div>
      <div style='font-family:"JetBrains Mono",monospace;font-size:0.62rem;
                  color:rgba(0,210,255,0.45);letter-spacing:2.5px;
                  text-transform:uppercase;margin-top:4px'>
        Injury Detection System
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Model")
    model_choice = st.selectbox(
        "YOLOv8 Weights",
        ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt"],
        index=2,
        help="Larger = more accurate, slower"
    )
    conf_thresh = st.slider("Detection Confidence", 0.25, 0.90, 0.50, 0.05)
    kp_conf     = st.slider("Keypoint Confidence",  0.20, 0.80, 0.40, 0.05)

    if st.button("⚡ Load Model"):
        with st.spinner("Loading weights..."):
            ok = load_model(model_choice)
        if ok:
            st.success(f"✅ {model_choice} ready")
        else:
            st.error("❌ Model load failed. Check ultralytics install.")

    st.markdown("---")
    st.markdown("### 🚨 Injury Thresholds")
    t_knee  = st.slider("Knee Hyperextension (°)",  155, 185, 170)
    t_elbow = st.slider("Elbow Hyperextension (°)", 160, 185, 175)
    t_hip   = st.slider("Hip Abduction (°)",         30,  80,  50)
    t_spine = st.slider("Spine Lateral Flex (°)",    20,  60,  40)

    thresholds = {"knee": t_knee, "elbow": t_elbow, "hip": t_hip, "spine": t_spine}

    st.markdown("---")
    st.markdown("### 🎥 Output")
    show_skeleton  = st.toggle("Draw Skeleton",    value=True)
    show_angles    = st.toggle("Show Joint Angles", value=True)
    show_bbox      = st.toggle("Show Bounding Box", value=True)
    save_report    = st.toggle("Save JSON Report",  value=True)
    process_every  = st.slider("Process Every N Frames", 1, 6, 2,
                                help="Skip frames for speed")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.62rem;
                color:rgba(255,255,255,0.2);line-height:1.8'>
      YOLOv8-Pose · COCO 17-pt<br>
      Real-time biomechanics<br>
      Rule-based risk classifier
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="pg-hero">
  <div class="pg-logo">🛡️ PoseGuard</div>
  <div class="pg-tagline">Real-time Athlete Injury Detection System</div>
  <div style="margin-top:16px">
    <span class="pg-badge">YOLOv8-Pose</span>
    <span class="pg-badge">Biomechanics Engine</span>
    <span class="pg-badge">Multi-Person Tracking</span>
    <span class="pg-badge">Risk Classification</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_analyze, tab_realtime, tab_report, tab_about = st.tabs([
    "📹  Analyze Video",
    "🔴  Realtime",
    "📊  Report",
    "ℹ️  About",
])


# ════════════════════════════════════════════════
# TAB 1 — ANALYZE VIDEO
# ════════════════════════════════════════════════
with tab_analyze:

    # ── Metric bar ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card cyan">
          <div class="metric-label">Frames Processed</div>
          <div class="metric-value cyan">{st.session_state.frame_count:,}</div>
          <div class="metric-sub">total frames</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card red">
          <div class="metric-label">Injury Events</div>
          <div class="metric-value red">{st.session_state.total_events}</div>
          <div class="metric-sub">total detected</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card amber">
          <div class="metric-label">High Risk</div>
          <div class="metric-value amber">{st.session_state.high_count}</div>
          <div class="metric-sub">critical events</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card green">
          <div class="metric-label">Medium Risk</div>
          <div class="metric-value green">{st.session_state.medium_count}</div>
          <div class="metric-sub">warning events</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Upload Video</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your sports video here",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if not st.session_state.model_loaded:
        st.info("💡 Load the model from the sidebar before running analysis.")

    if uploaded and st.session_state.model_loaded:
        model = st.session_state.yolo_model

        col_run, col_reset = st.columns([1, 5])
        with col_run:
            run_btn = st.button("▶  Run Analysis", use_container_width=True)
        with col_reset:
            if st.button("↺  Reset", use_container_width=False):
                for k in ["event_log","frame_count","total_events",
                          "high_count","medium_count","low_count",
                          "risk_timeline","person_count_history"]:
                    st.session_state[k] = [] if isinstance(st.session_state[k], list) else 0
                st.rerun()

        if run_btn:
            # Reset state
            st.session_state.event_log = []
            st.session_state.frame_count = 0
            st.session_state.total_events = 0
            st.session_state.high_count = 0
            st.session_state.medium_count = 0
            st.session_state.low_count = 0
            st.session_state.risk_timeline = []
            st.session_state.person_count_history = []

            # Write to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30

            # UI layout while processing
            st.markdown("<div class='section-header'>Live Preview</div>", unsafe_allow_html=True)
            c_vid, c_events = st.columns([3, 2])

            with c_vid:
                frame_display = st.empty()
                prog_bar = st.progress(0)
                status_txt = st.empty()

            with c_events:
                st.markdown("""
                <div style='font-family:"JetBrains Mono",monospace;font-size:0.68rem;
                            color:rgba(0,210,255,0.55);letter-spacing:2px;
                            text-transform:uppercase;margin-bottom:12px'>
                  Live Events
                </div>""", unsafe_allow_html=True)
                event_container = st.empty()

            frame_idx = 0
            event_log = []
            t_start = time.time()
            recent_events: deque = deque(maxlen=8)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % process_every != 0:
                    continue

                st.session_state.frame_count += 1
                progress = min(frame_idx / max(total_frames, 1), 1.0)
                elapsed = time.time() - t_start
                proc_fps = st.session_state.frame_count / max(elapsed, 0.001)

                # ── YOLO inference ──
                results = model(frame, conf=conf_thresh, verbose=False)
                frame_events = []
                person_count = 0

                for result in results:
                    if result.keypoints is None:
                        continue
                    kp_data = result.keypoints.data.cpu().numpy()
                    boxes   = result.boxes.xyxy.cpu().numpy()

                    for i in range(len(boxes)):
                        person_count += 1
                        kps = extract_keypoints(kp_data[i], kp_conf)
                        angles = compute_angles(kps)
                        events = classify_injuries(angles, kps, thresholds, frame_idx, i)

                        risk_score = max((e["score"] for e in events), default=0)
                        level = risk_level(risk_score)
                        color = get_risk_color_cv(level if events else "NONE")

                        if show_skeleton:
                            frame = draw_skeleton_on_frame(frame, kps, color)

                        if show_bbox:
                            x1,y1,x2,y2 = [int(v) for v in boxes[i]]
                            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
                            cv2.putText(frame,f"ID:{i+1}",(x1,y1-8),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv2.LINE_AA)

                        if show_angles:
                            angle_joints = ["left_knee","right_knee","left_elbow","right_elbow"]
                            kp_map = {j:j for j in angle_joints}
                            for jnt in angle_joints:
                                angle = angles.get(jnt)
                                if angle and kps.get(jnt):
                                    x,y = int(kps[jnt][0])+8, int(kps[jnt][1])
                                    cv2.putText(frame,f"{angle:.0f}°",(x,y),
                                                cv2.FONT_HERSHEY_SIMPLEX,0.40,
                                                (255,220,80),1,cv2.LINE_AA)

                        for ev in events:
                            cv2.putText(frame, f"⚠ {ev['label'][:22]}",
                                        (int(boxes[i][0]), int(boxes[i][1])-26),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        get_risk_color_cv(ev["risk_level"]), 1, cv2.LINE_AA)

                        frame_events.extend(events)

                # HUD overlay
                overlay = frame.copy()
                cv2.rectangle(overlay,(0,0),(300,70),(0,0,0),-1)
                cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)
                cv2.putText(frame,"POSEGUARD",(8,18),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,210,255),2,cv2.LINE_AA)
                cv2.putText(frame,f"Frame:{frame_idx}  FPS:{proc_fps:.1f}  Persons:{person_count}",
                            (8,38),cv2.FONT_HERSHEY_SIMPLEX,0.40,(180,180,180),1,cv2.LINE_AA)
                cv2.putText(frame,f"Events:{st.session_state.total_events}",
                            (8,58),cv2.FONT_HERSHEY_SIMPLEX,0.40,
                            (50,60,255) if st.session_state.total_events else (0,200,100),
                            1,cv2.LINE_AA)

                # Update state
                for ev in frame_events:
                    st.session_state.total_events += 1
                    if ev["risk_level"] == "HIGH":   st.session_state.high_count   += 1
                    elif ev["risk_level"] == "MEDIUM": st.session_state.medium_count += 1
                    else: st.session_state.low_count += 1
                    event_log.append(ev)
                    recent_events.appendleft(ev)

                st.session_state.risk_timeline.append(
                    (frame_idx, max((e["score"] for e in frame_events), default=0))
                )
                st.session_state.person_count_history.append((frame_idx, person_count))

                # Display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display.image(frame_rgb, channels="RGB", use_container_width=True)
                prog_bar.progress(progress)
                status_txt.markdown(
                    f'<div class="status-live"><div class="status-dot"></div>'
                    f'Processing · Frame {frame_idx}/{total_frames} · {proc_fps:.1f} FPS</div>',
                    unsafe_allow_html=True
                )

                # Live event list
                if recent_events:
                    html = ""
                    for ev in list(recent_events)[:6]:
                        c = get_risk_color_hex(ev["risk_level"])
                        html += f"""
                        <div style='background:#07111f;border-left:3px solid {c};
                                    border-radius:0 8px 8px 0;padding:10px 14px;
                                    margin-bottom:6px'>
                          <div style='font-weight:600;font-size:0.78rem;color:#f0f4ff'>
                            {ev["label"]}
                          </div>
                          <div style='font-family:"JetBrains Mono",monospace;font-size:0.62rem;
                                      color:rgba(255,255,255,0.35);margin-top:2px'>
                            Frame {ev["frame"]} · Track {ev["track_id"]+1} · Score {ev["score"]:.0%}
                          </div>
                        </div>"""
                    event_container.markdown(html, unsafe_allow_html=True)

            cap.release()
            st.session_state.event_log = event_log
            status_txt.markdown(
                '<div class="status-live" style="background:rgba(0,210,255,0.08);'
                'border-color:rgba(0,210,255,0.3);color:#00d2ff">'
                '✅ Analysis Complete</div>',
                unsafe_allow_html=True
            )

            if save_report and event_log:
                Path("outputs/logs").mkdir(parents=True, exist_ok=True)
                with open("outputs/logs/poseguard_report.json","w") as f:
                    json.dump(event_log, f, indent=2)

            # ── Risk timeline chart ──
            if st.session_state.risk_timeline:
                st.markdown("<div class='section-header'>Risk Score Timeline</div>",
                            unsafe_allow_html=True)
                frames_t = [x[0] for x in st.session_state.risk_timeline]
                scores_t = [x[1] for x in st.session_state.risk_timeline]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=frames_t, y=scores_t,
                    mode="lines",
                    line=dict(color="#00d2ff", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(0,210,255,0.07)",
                    name="Risk Score"
                ))
                fig.add_hline(y=0.75, line_dash="dash",
                              line_color="rgba(255,60,80,0.5)",
                              annotation_text="HIGH",
                              annotation_font_color="#ff3c50")
                fig.add_hline(y=0.45, line_dash="dash",
                              line_color="rgba(255,176,32,0.5)",
                              annotation_text="MEDIUM",
                              annotation_font_color="#ffb020")
                fig.update_layout(
                    paper_bgcolor="#050810", plot_bgcolor="#070c1a",
                    font=dict(family="JetBrains Mono", color="#aab4c8", size=11),
                    margin=dict(l=40,r=20,t=20,b=40),
                    height=220,
                    xaxis=dict(title="Frame", gridcolor="rgba(255,255,255,0.05)",
                               showline=False),
                    yaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.05)",
                               range=[0,1], showline=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    elif uploaded and not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model from the sidebar first.")

    elif not uploaded:
        st.markdown("""
        <div class="upload-zone">
          <span class="upload-icon">🎬</span>
          <div class="upload-title">Drop a sports video to begin</div>
          <div class="upload-sub">MP4 · AVI · MOV · MKV · up to 500 MB</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# TAB 2 — REALTIME
# ════════════════════════════════════════════════
with tab_realtime:
    st.markdown("""
    <div style='background:#070c1a;border:1px solid rgba(0,210,255,0.12);
                border-radius:14px;padding:28px 32px;margin-bottom:20px'>
      <div style='font-size:1.1rem;font-weight:700;color:#f0f4ff;margin-bottom:8px'>
        🔴 Real-time Camera Feed
      </div>
      <div style='font-family:"JetBrains Mono",monospace;font-size:0.72rem;
                  color:rgba(255,255,255,0.35);line-height:1.8'>
        Supports webcam, RTSP, and USB camera sources.<br>
        Load the model from the sidebar, then press Start.
      </div>
    </div>
    """, unsafe_allow_html=True)

    cam_source = st.text_input("Camera Source", value="0",
                               help="0 = default webcam, or paste RTSP URL")
    rt_col1, rt_col2 = st.columns(2)

    with rt_col1:
        start_rt = st.button("▶  Start Feed", use_container_width=True)
    with rt_col2:
        stop_rt = st.button("⏹  Stop", use_container_width=True)

    if start_rt:
        if not st.session_state.model_loaded:
            st.error("Load model first.")
        else:
            model = st.session_state.yolo_model
            source = int(cam_source) if cam_source.isdigit() else cam_source
            cap = cv2.VideoCapture(source)

            rt_frame   = st.empty()
            rt_status  = st.empty()
            rt_events  = st.empty()
            recent_rt  = deque(maxlen=6)
            frame_idx  = 0
            t_start    = time.time()

            while not stop_rt:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % process_every != 0:
                    continue

                results = model(frame, conf=conf_thresh, verbose=False)
                for result in results:
                    if result.keypoints is None:
                        continue
                    kp_data = result.keypoints.data.cpu().numpy()
                    boxes   = result.boxes.xyxy.cpu().numpy()
                    for i in range(len(boxes)):
                        kps    = extract_keypoints(kp_data[i], kp_conf)
                        angles = compute_angles(kps)
                        events = classify_injuries(angles, kps, thresholds, frame_idx, i)
                        risk   = max((e["score"] for e in events), default=0)
                        level  = risk_level(risk)
                        color  = get_risk_color_cv(level if events else "NONE")
                        if show_skeleton:
                            frame = draw_skeleton_on_frame(frame, kps, color)
                        if show_bbox:
                            x1,y1,x2,y2 = [int(v) for v in boxes[i]]
                            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                        for ev in events:
                            recent_rt.appendleft(ev)

                fps_rt = frame_idx / max(time.time()-t_start, 0.001)
                cv2.putText(frame,f"PoseGuard LIVE | {fps_rt:.1f}fps",
                            (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,210,255),2)
                rt_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)

                if recent_rt:
                    html = "".join([
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;'
                        f'color:{get_risk_color_hex(e["risk_level"])};padding:3px 0">'
                        f'⚠ {e["label"]} · {e["score"]:.0%}</div>'
                        for e in list(recent_rt)[:4]
                    ])
                    rt_events.markdown(html, unsafe_allow_html=True)

            cap.release()


# ════════════════════════════════════════════════
# TAB 3 — REPORT
# ════════════════════════════════════════════════
with tab_report:
    st.markdown("<div class='section-header'>Analysis Report</div>", unsafe_allow_html=True)

    events = st.session_state.event_log

    if not events:
        st.markdown("""
        <div style='background:#070c1a;border:1px solid rgba(255,255,255,0.06);
                    border-radius:14px;padding:40px;text-align:center;color:rgba(255,255,255,0.25);
                    font-family:"JetBrains Mono",monospace;font-size:0.78rem;letter-spacing:1.5px'>
          NO DATA YET — RUN AN ANALYSIS FIRST
        </div>""", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(events)

        # ── Summary row ──
        r1, r2, r3, r4 = st.columns(4)
        counts = df["risk_level"].value_counts()
        with r1:
            st.markdown(f'<div class="metric-card cyan"><div class="metric-label">Total Events</div>'
                        f'<div class="metric-value cyan">{len(df)}</div></div>',
                        unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card red"><div class="metric-label">High Risk</div>'
                        f'<div class="metric-value red">{counts.get("HIGH",0)}</div></div>',
                        unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card amber"><div class="metric-label">Medium Risk</div>'
                        f'<div class="metric-value amber">{counts.get("MEDIUM",0)}</div></div>',
                        unsafe_allow_html=True)
        with r4:
            st.markdown(f'<div class="metric-card green"><div class="metric-label">Low Risk</div>'
                        f'<div class="metric-value green">{counts.get("LOW",0)}</div></div>',
                        unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        c_pie, c_bar = st.columns(2)

        with c_pie:
            st.markdown("<div class='section-header'>Injury Type Distribution</div>",
                        unsafe_allow_html=True)
            type_counts = df["label"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.55,
                marker=dict(colors=["#00d2ff","#ff3c50","#ffb020","#00e87a",
                                     "#a855f7","#f97316","#ec4899"]),
                textfont=dict(family="JetBrains Mono", size=11),
            ))
            fig_pie.update_layout(
                paper_bgcolor="#050810", plot_bgcolor="#050810",
                font=dict(family="JetBrains Mono", color="#aab4c8"),
                margin=dict(l=10,r=10,t=10,b=10), height=260,
                legend=dict(font=dict(size=10)),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_bar:
            st.markdown("<div class='section-header'>Events by Risk Level</div>",
                        unsafe_allow_html=True)
            risk_order = ["HIGH", "MEDIUM", "LOW"]
            risk_vals  = [counts.get(r, 0) for r in risk_order]
            fig_bar = go.Figure(go.Bar(
                x=risk_order, y=risk_vals,
                marker_color=["#ff3c50","#ffb020","#00e87a"],
                text=risk_vals,
                textposition="auto",
                textfont=dict(family="JetBrains Mono", size=13, color="#050810"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="#050810", plot_bgcolor="#070c1a",
                font=dict(family="JetBrains Mono", color="#aab4c8"),
                margin=dict(l=20,r=20,t=10,b=20), height=260,
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showline=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showline=False),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Event table ──
        st.markdown("<div class='section-header'>Full Event Log</div>", unsafe_allow_html=True)

        # Color-coded risk column
        def style_risk(val):
            colors = {"HIGH": "#ff3c50", "MEDIUM": "#ffb020", "LOW": "#00e87a"}
            c = colors.get(val, "#aaa")
            return f"color: {c}; font-weight: 600; font-family: JetBrains Mono, monospace"

        styled = (
            df[["frame","track_id","label","risk_level","score","detail"]]
            .rename(columns={
                "frame":"Frame","track_id":"Track","label":"Injury Type",
                "risk_level":"Risk","score":"Score","detail":"Detail"
            })
            .style
            .applymap(style_risk, subset=["Risk"])
            .format({"Score": "{:.0%}"})
            .set_properties(**{
                "font-family": "JetBrains Mono, monospace",
                "font-size": "0.75rem",
            })
        )
        st.dataframe(styled, use_container_width=True, height=320)

        # ── Download ──
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "⬇  Download JSON Report",
                data=json.dumps(events, indent=2),
                file_name="poseguard_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "⬇  Download CSV",
                data=df.to_csv(index=False),
                file_name="poseguard_report.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div style='max-width:780px'>

    <div style='font-size:1.6rem;font-weight:800;color:#f0f4ff;
                letter-spacing:-0.5px;margin-bottom:6px'>
      PoseGuard — Athlete Injury Detection
    </div>
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.72rem;
                color:rgba(0,210,255,0.55);letter-spacing:2px;margin-bottom:28px'>
      YOLOV8-POSE · BIOMECHANICS ENGINE · RISK CLASSIFIER
    </div>

    <div style='background:#070c1a;border:1px solid rgba(0,210,255,0.10);
                border-radius:14px;padding:28px 32px;margin-bottom:18px'>
      <div style='font-weight:700;font-size:0.9rem;color:#00d2ff;margin-bottom:12px'>
        🔬 How It Works
      </div>
      <div style='font-size:0.82rem;line-height:1.9;color:rgba(255,255,255,0.65)'>
        <b style='color:#f0f4ff'>1. Detection</b> — YOLOv8-Pose detects all persons in each frame
        and extracts 17 COCO keypoints per person with confidence scores.<br>
        <b style='color:#f0f4ff'>2. Biomechanics</b> — Joint angles are computed at knees, hips,
        elbows, shoulders, and spine using vector math on keypoint coordinates.<br>
        <b style='color:#f0f4ff'>3. Temporal Smoothing</b> — A sliding window averages keypoint
        positions over N frames to reduce detection noise.<br>
        <b style='color:#f0f4ff'>4. Risk Classification</b> — Rule-based thresholds fire injury
        events when angles exceed safe biomechanical ranges.<br>
        <b style='color:#f0f4ff'>5. Tracking</b> — IoU-based tracking assigns consistent IDs
        across frames for per-athlete event history.
      </div>
    </div>

    <div style='background:#070c1a;border:1px solid rgba(255,60,80,0.12);
                border-radius:14px;padding:28px 32px;margin-bottom:18px'>
      <div style='font-weight:700;font-size:0.9rem;color:#ff3c50;margin-bottom:14px'>
        🚨 Injury Detection Rules
      </div>
      <table style='width:100%;border-collapse:collapse;font-size:0.78rem;
                    font-family:"JetBrains Mono",monospace'>
        <tr style='color:rgba(0,210,255,0.6);border-bottom:1px solid rgba(255,255,255,0.06)'>
          <th style='text-align:left;padding:8px 0;'>Signal</th>
          <th style='text-align:left;padding:8px 0;'>Threshold</th>
          <th style='text-align:left;padding:8px 0;'>Injury Type</th>
        </tr>
        <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
          <td style='padding:8px 0'>Knee angle</td>
          <td style='color:#ffb020'>&gt; 170°</td>
          <td style='color:#ff3c50'>ACL / Ligament Risk</td>
        </tr>
        <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
          <td style='padding:8px 0'>Elbow angle</td>
          <td style='color:#ffb020'>&gt; 175°</td>
          <td style='color:#ff3c50'>Elbow Hyperextension</td>
        </tr>
        <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
          <td style='padding:8px 0'>Hip abduction</td>
          <td style='color:#ffb020'>&gt; 50°</td>
          <td style='color:#ff3c50'>Hamstring Strain</td>
        </tr>
        <tr style='color:rgba(255,255,255,0.6);border-bottom:1px solid rgba(255,255,255,0.04)'>
          <td style='padding:8px 0'>Spine lateral flex</td>
          <td style='color:#ffb020'>&gt; 40°</td>
          <td style='color:#ff3c50'>Spine / Back Stress</td>
        </tr>
        <tr style='color:rgba(255,255,255,0.6)'>
          <td style='padding:8px 0'>Rapid torso drop</td>
          <td style='color:#ffb020'>6-frame window</td>
          <td style='color:#ff3c50'>Fall Detected</td>
        </tr>
      </table>
    </div>

    <div style='background:#070c1a;border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:28px 32px'>
      <div style='font-weight:700;font-size:0.9rem;color:#f0f4ff;margin-bottom:12px'>
        🛠 Tech Stack
      </div>
      <div style='display:flex;gap:10px;flex-wrap:wrap'>
        {''.join([f'<span style="background:rgba(0,210,255,0.08);border:1px solid rgba(0,210,255,0.2);color:#00d2ff;font-family:JetBrains Mono,monospace;font-size:0.68rem;padding:5px 14px;border-radius:100px">{t}</span>' for t in ["YOLOv8-Pose","Streamlit","OpenCV","NumPy","Plotly","PyTorch","Ultralytics"]])}
      </div>
    </div>

    </div>
    """, unsafe_allow_html=True)