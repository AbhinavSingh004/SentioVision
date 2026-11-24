# app.py
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image
from ml_engine import predict_emotion
from utils import FPSCounter, create_emotion_dataframe, format_log_entry, load_asset

st.set_page_config(page_title="SentioVision AI", layout="wide")

# 1. Alag files se HTML, CSS aur JS load karna
css_code = load_asset("assets/style.css")
html_code = load_asset("assets/header.html")
js_code = load_asset("assets/script.js")

st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)
components.html(f"{html_code}<script>{js_code}</script>", height=105)

# Sidebar Controls
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Input Mode Chunein:", ["Image Upload (Fast)", "Live Webcam"])

if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []

# --- MODE 1: IMAGE UPLOAD ---
if mode == "Image Upload (Fast)":
    uploaded_file = st.file_uploader("Koi bhi face photo upload karein", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with st.spinner("ML Model emotion analyze kar raha hai..."):
            processed_frame, detections = predict_emotion(frame)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Analyzed Output", use_container_width=True)
        
        with col2:
            st.subheader("Detection Results")
            if detections:
                for idx, d in enumerate(detections):
                    st.markdown(f"""
                    <div class="emotion-card">
                        <h4 style="margin:0; color:#38bdf8;">Face #{idx+1}: {d['emotion'].upper()}</h4>
                        <p style="margin:4px 0 0 0; color:#cbd5e1;">Confidence Score: <b>{d['confidence']}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.bar_chart(d['all_scores'])
            else:
                st.warning("Koi face detect nahi hua.")

# --- MODE 2: LIVE WEBCAM ---
elif mode == "Live Webcam":
    run_cam = st.sidebar.checkbox("Camera Start Karein")
    col1, col2 = st.columns([2, 1])
    frame_window = col1.image([])
    info_box = col2.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        fps_counter = FPSCounter()

        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam access nahi ho pa raha hai.")
                break

            processed_frame, detections = predict_emotion(frame)
            fps = fps_counter.get_fps()

            if detections:
                top_emotion = detections[0]['emotion']
                conf = detections[0]['confidence']
                st.session_state.emotion_log.append(format_log_entry(top_emotion, conf))
                
                info_box.markdown(f"""
                <div class="emotion-card">
                    <span class="status-badge">FPS: {fps}</span>
                    <h2 style="margin:10px 0 5px 0; color:#38bdf8;">{top_emotion.upper()}</h2>
                    <p style="margin:0; color:#94a3b8;">Confidence: <b>{conf}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                info_box.markdown(f"""
                <div class="emotion-card">
                    <span class="status-badge">FPS: {fps}</span>
                    <p style="margin-top:10px; color:#94a3b8;">Scanning for faces...</p>
                </div>
                """, unsafe_allow_html=True)

            frame_window.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

        cap.release()

# --- ANALYTICS SECTION ---
st.divider()
st.subheader("Session Emotion Analytics")
if st.session_state.emotion_log:
    df = create_emotion_dataframe(st.session_state.emotion_log)
    st.dataframe(df.tail(5), use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report (CSV)", data=csv, file_name="sentiovision_report.csv", mime="text/csv")
else:
    st.info("Abhi tak koi emotion record nahi hua hai.")
    # --- DASHBOARD FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; padding: 12px; color: #94a3b8; font-size: 14px;">
    Designed by <b style="color: #38bdf8;">Abhinav</b> | Student of <b>MCA (AI & Data Science)</b> in <b>Rayat Bahra University</b>
</div>
""", unsafe_allow_html=True)