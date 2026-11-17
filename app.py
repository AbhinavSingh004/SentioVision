import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="SentioVision AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 SentioVision: Real-Time Facial Emotion & Mood Analytics")
st.markdown("Live Computer Vision & Deep Learning Dashboard powered by **DeepFace (CNN)** & **OpenCV**")
st.markdown("---")

st.sidebar.header("⚙️ Control Panel")
run_camera = st.sidebar.checkbox("📷 Start Live Webcam", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Analytics Data")

if "emotion_counts" not in st.session_state:
    st.session_state.emotion_counts = {
        "happy": 0, "sad": 0, "angry": 0, "neutral": 0, "surprise": 0, "fear": 0, "disgust": 0
    }

if "session_logs" not in st.session_state:
    st.session_state.session_logs = []

if st.sidebar.button("🔄 Reset Analytics"):
    st.session_state.emotion_counts = {k: 0 for k in st.session_state.emotion_counts}
    st.session_state.session_logs = []
    st.sidebar.success("Analytics data reset!")

if len(st.session_state.session_logs) > 0:
    log_df = pd.DataFrame(st.session_state.session_logs)
    csv_data = log_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📄 Download Session Report (CSV)",
        data=csv_data,
        file_name=f"SentioVision_Report_{datetime.now().strftime('%H%M%S')}.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Chehre par achhi lighting rakhein taaki ML model sahi prediction kar sake.")

recommendations = {
    "happy": ("🌟 Great Energy Detected!", "Aapka mood kaafi positive hai. Creative ya productive kaam karne ke liye yeh best time hai."),
    "sad": ("💙 Cheer Up!", "Thoda low feel ho raha hai? 5 minute ka break lein, paani piyein ya apna favorite music sunein."),
    "angry": ("🧘 Take a Deep Breath", "Stress ya gussa detect hua hai. Screen se thodi der door hatein aur relax karein."),
    "surprise": ("😲 Something Unexpected?", "Lagta hai kisi cheez ne aapka dhyan achanak khinch liya hai!"),
    "neutral": ("🎯 Focused & Calm", "Aapka mind bilkul shaant aur balanced hai. Coding aur padhai ke liye yeh ideal state hai."),
    "fear": ("🛡️ Stay Calm", "Sab kuch control mein hai. Gehri saans lein aur relax rahein."),
    "disgust": ("🤔 Uncomfortable?", "Aas-paas ka mahol thoda badalne ki koshish karein.")
}

col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("🎥 Live AI Vision Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Live Mood & Confidence Metrics")
    m1, m2 = st.columns(2)
    emotion_metric = m1.empty()
    confidence_metric = m2.empty()
    
    st.markdown("---")
    recommendation_box = st.empty()
    
    st.markdown("### 📈 Real-Time Emotion Distribution")
    chart_placeholder = st.empty()

if run_camera:
    # DirectShow backend (cv2.CAP_DSHOW) Windows par camera ko turant kholta hai
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    frame_count = 0
    dominant_emotion = "neutral"
    confidence = 0.0
    box = None
    
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Webcam open nahi ho pa raha! Check karein ki koi aur app (Zoom/Camera) camera use to nahi kar raha.")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Pehle camera dikhao, har 8th frame par fast Emotion AI chalao
        if frame_count % 8 == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if results:
                    res = results[0]
                    dominant_emotion = res['dominant_emotion']
                    confidence = round(res['emotion'][dominant_emotion], 1)
                    
                    if dominant_emotion in st.session_state.emotion_counts:
                        st.session_state.emotion_counts[dominant_emotion] += 1
                    
                    st.session_state.session_logs.append({
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Emotion": dominant_emotion,
                        "Confidence_%": confidence
                    })
                    
                    region = res['region']
                    box = (region['x'], region['y'], region['w'], region['h'])
                    
            except Exception:
                pass
        
        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 127), 2)
            label = f"{dominant_emotion.upper()} ({confidence}%)"
            cv2.putText(frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 127), 2)
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        emotion_metric.metric("Detected Mood", dominant_emotion.capitalize())
        confidence_metric.metric("AI Confidence", f"{confidence}%")
        
        title, msg = recommendations.get(dominant_emotion, ("🤖 Scanning...", "Analyzing facial features..."))
        recommendation_box.info(f"**{title}**\n\n{msg}")
        
        df_emotions = pd.DataFrame(
            list(st.session_state.emotion_counts.items()),
            columns=["Emotion", "Frames Detected"]
        ).set_index("Emotion")
        chart_placeholder.bar_chart(df_emotions)
        
    cap.release()
else:
    frame_placeholder.info("👈 Project shuru karne ke liye left sidebar mein **'Start Live Webcam'** par tick karein.")