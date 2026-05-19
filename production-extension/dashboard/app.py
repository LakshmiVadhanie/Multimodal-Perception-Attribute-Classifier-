"""
dashboard/app.py

Streamlit dashboard for road user attribute analysis.

Two modes:
  1. Upload Mode  — drag in an image or video, see detections + alerts
  2. Live Mode    — point at an RTSP stream URL, watch alerts come in real-time

Run:
    streamlit run dashboard/app.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

SEVERITY_COLOR = {
    "critical": "#FF2D55",
    "high": "#FF6B35",
    "medium": "#FFB347",
    "low": "#4ECDC4",
}

ATTR_ICONS = {
    "mobility": "movement",
    "orientation": "compass",
    "occlusion": "eye-off",
    "lighting": "sun",
    "size": "maximize-2",
    "posture": "user",
    "group": "users",
    "attention": "alert-circle",
}

st.set_page_config(
    page_title="Road User Monitor",
    page_icon="traffic_light",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f0f0f;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .alert-card {
        border-left: 3px solid;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
        background: #fafafa;
    }

    .alert-high   { border-color: #FF6B35; background: #fff5f0; }
    .alert-critical { border-color: #FF2D55; background: #fff0f3; }
    .alert-medium { border-color: #FFB347; background: #fffbf0; }
    .alert-low    { border-color: #4ECDC4; background: #f0fffe; }

    .alert-title {
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 0.2rem;
    }

    .alert-meta {
        font-size: 0.75rem;
        color: #666;
        font-family: 'Space Mono', monospace;
    }

    .attr-pill {
        display: inline-block;
        background: #f0f0f0;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin: 2px;
        font-family: 'Space Mono', monospace;
        color: #333;
    }

    .detection-card {
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        background: white;
    }

    .user-type-badge {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 2px 8px;
        border-radius: 3px;
        color: white;
    }

    .badge-vehicle    { background: #2D5BE3; }
    .badge-pedestrian { background: #1AAB74; }
    .badge-cyclist    { background: #9B59B6; }

    .stat-number {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f0f0f;
        line-height: 1;
    }

    .stat-label {
        font-size: 0.72rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-ok      { background: #1AAB74; }
    .dot-error   { background: #FF2D55; }
    .dot-waiting { background: #FFB347; animation: pulse 1.5s infinite; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.3; }
    }

    .stButton > button {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        border-radius: 4px;
        border: 1.5px solid #0f0f0f;
        background: #0f0f0f;
        color: white;
        padding: 0.5rem 1.2rem;
    }

    .stButton > button:hover {
        background: #333;
        border-color: #333;
    }

    div[data-testid="stSidebarContent"] {
        background: #0f0f0f;
        color: white;
    }

    div[data-testid="stSidebarContent"] label {
        color: #ccc !important;
    }

    div[data-testid="stSidebarContent"] .stSelectbox label,
    div[data-testid="stSidebarContent"] .stTextInput label {
        color: #aaa !important;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def poll_job(job_id: str) -> Dict:
    r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=5)
    return r.json()


def fetch_alerts(severity: Optional[str] = None, road_user_type: Optional[str] = None) -> List[Dict]:
    params = {}
    if severity:
        params["severity"] = severity
    if road_user_type:
        params["road_user_type"] = road_user_type
    r = requests.get(f"{API_BASE}/alerts", params=params, timeout=5)
    return r.json().get("alerts", [])


def render_alert_card(alert: Dict):
    sev = alert.get("severity", "low")
    cls = f"alert-{sev}"
    attrs_html = "".join(
        f'<span class="attr-pill">{k}: {v}</span>'
        for k, v in alert.get("attributes", {}).items()
        if v
    )
    ts = alert.get("timestamp_readable", "")
    st.markdown(f"""
    <div class="alert-card {cls}">
        <div class="alert-title">{alert['rule_name']}</div>
        <div class="alert-meta">
            track #{alert['track_id']} &nbsp;·&nbsp; {alert['road_user_type']} &nbsp;·&nbsp; {ts}
        </div>
        <div style="margin-top:6px">{attrs_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_detection_card(det: Dict):
    rut = det.get("road_user_type", "vehicle")
    badge_cls = f"badge-{rut}"
    conf = round(det.get("confidence", 0) * 100)
    attrs = det.get("attributes", {})
    attrs_html = "".join(
        f'<span class="attr-pill">{k}: {v}</span>'
        for k, v in attrs.items()
        if v
    )
    alert_count = len(det.get("alerts", []))
    alert_badge = (
        f'<span style="color:#FF6B35;font-size:0.78rem;font-weight:600">⚠ {alert_count} alert{"s" if alert_count != 1 else ""}</span>'
        if alert_count else ""
    )
    st.markdown(f"""
    <div class="detection-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <span class="user-type-badge {badge_cls}">{rut}</span>
            <span style="font-size:0.78rem;color:#888">confidence: {conf}%</span>
            {alert_badge}
        </div>
        <div>{attrs_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div style="color:white;font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;margin-bottom:0.2rem">ROAD USER<br>MONITOR</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#555;font-size:0.72rem;margin-bottom:2rem">v2.0 · Attribute Analysis</div>', unsafe_allow_html=True)

    api_ok = check_api()
    dot_cls = "dot-ok" if api_ok else "dot-error"
    status_txt = "API connected" if api_ok else "API offline"
    st.markdown(f'<div style="color:#aaa;font-size:0.78rem;margin-bottom:1.5rem"><span class="status-dot {dot_cls}"></span>{status_txt}</div>', unsafe_allow_html=True)

    mode = st.selectbox("MODE", ["Upload Image", "Upload Video", "Live Stream", "Alert Log"])

    st.markdown("---")
    st.markdown('<div style="color:#555;font-size:0.7rem;font-family:Space Mono,monospace">FILTER ALERTS</div>', unsafe_allow_html=True)
    sev_filter = st.selectbox("Severity", ["All", "critical", "high", "medium", "low"])
    rut_filter = st.selectbox("Road User", ["All", "vehicle", "pedestrian", "cyclist"])


# ── main content ───────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">Road User Attribute Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detection · Classification · Alerting</div>', unsafe_allow_html=True)

if not api_ok:
    st.error("Cannot reach the analysis API at `http://localhost:8000`. Start it with: `uvicorn serving.api_v2:app --port 8000`")
    st.stop()


# ── upload image ───────────────────────────────────────────────────────────────
if mode == "Upload Image":
    uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        col_img, col_results = st.columns([1, 1], gap="large")

        with col_img:
            st.image(uploaded, caption="Uploaded image", use_column_width=True)

        with col_results:
            with st.spinner("Analyzing..."):
                r = requests.post(
                    f"{API_BASE}/analyze/image",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )

            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
            else:
                data = r.json()
                dets = data.get("detections", [])
                total_alerts = data.get("total_alerts", 0)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="stat-number">{len(dets)}</div><div class="stat-label">Detections</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="stat-number">{total_alerts}</div><div class="stat-label">Alerts</div>', unsafe_allow_html=True)
                with c3:
                    types = {d["road_user_type"] for d in dets}
                    st.markdown(f'<div class="stat-number">{len(types)}</div><div class="stat-label">Types</div>', unsafe_allow_html=True)

                st.markdown("---")

                if not dets:
                    st.info("No road users detected in this image.")
                else:
                    for det in dets:
                        render_detection_card(det)
                        for alert in det.get("alerts", []):
                            render_alert_card(alert)


# ── upload video ───────────────────────────────────────────────────────────────
elif mode == "Upload Video":
    uploaded = st.file_uploader("Drop a video file here", type=["mp4", "avi", "mov"])
    if uploaded:
        with st.spinner("Submitting video for analysis..."):
            r = requests.post(
                f"{API_BASE}/analyze/video",
                files={"file": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                timeout=30,
            )

        if r.status_code != 200:
            st.error(f"Submission failed: {r.text}")
        else:
            job_id = r.json()["job_id"]
            st.success(f"Job submitted: `{job_id}`")

            progress_bar = st.progress(0)
            status_area = st.empty()
            result_area = st.empty()

            for i in range(120):  # poll for up to 2 minutes
                time.sleep(2)
                job = poll_job(job_id)
                status = job.get("status")

                if status == "running":
                    progress_bar.progress(min(0.9, i / 60))
                    status_area.info("Processing video...")
                elif status == "complete":
                    progress_bar.progress(1.0)
                    status_area.success("Analysis complete")
                    result = job.get("result", {})
                    with result_area.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Frames processed", result.get("total_frames_processed", 0))
                        c2.metric("Detections", result.get("total_detections", 0))
                        c3.metric("Classifications", result.get("total_classifications", 0))
                        c4.metric("Alerts", result.get("total_alerts", 0))
                        st.markdown("### Alerts")
                        for alert in result.get("alerts", []):
                            render_alert_card(alert)
                    break
                elif status == "failed":
                    progress_bar.progress(0)
                    status_area.error(f"Job failed: {job.get('error')}")
                    break


# ── live stream ────────────────────────────────────────────────────────────────
elif mode == "Live Stream":
    st.markdown("Point the pipeline at an RTSP camera or local stream.")
    stream_url = st.text_input("Stream URL", placeholder="rtsp://192.168.1.100:554/stream or /path/to/video.mp4")

    if stream_url:
        st.info(
            "The live stream is processed server-side. "
            "Alerts will appear in the **Alert Log** tab as they are generated. "
            "Refresh that tab periodically, or connect via the SSE endpoint:\n\n"
            f"`GET {API_BASE}/stream/alerts?source={stream_url}`"
        )
        st.code(f"""
# Python example — receive live alerts
import requests, json
url = "{API_BASE}/stream/alerts?source={stream_url}"
with requests.get(url, stream=True) as r:
    for line in r.iter_lines():
        if line.startswith(b"data:"):
            event = json.loads(line[5:])
            if event["alerts"]:
                print(event)
        """, language="python")


# ── alert log ──────────────────────────────────────────────────────────────────
elif mode == "Alert Log":
    col_controls, col_refresh = st.columns([3, 1])
    with col_refresh:
        if st.button("Refresh"):
            st.rerun()

    sev = None if sev_filter == "All" else sev_filter
    rut = None if rut_filter == "All" else rut_filter

    try:
        alerts = fetch_alerts(severity=sev, road_user_type=rut)
    except Exception as e:
        st.error(f"Could not fetch alerts: {e}")
        alerts = []

    if not alerts:
        st.info("No alerts recorded yet. Analyze an image or video first.")
    else:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for a in alerts:
            counts[a.get("severity", "low")] = counts.get(a.get("severity", "low"), 0) + 1

        c1, c2, c3, c4 = st.columns(4)
        for col, sev_key, label in [
            (c1, "critical", "Critical"),
            (c2, "high", "High"),
            (c3, "medium", "Medium"),
            (c4, "low", "Low"),
        ]:
            col.markdown(
                f'<div class="stat-number" style="color:{SEVERITY_COLOR[sev_key]}">{counts[sev_key]}</div>'
                f'<div class="stat-label">{label}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        for alert in reversed(alerts):
            render_alert_card(alert)
