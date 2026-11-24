# utils.py
import time
import pandas as pd
from datetime import datetime

class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()

    def get_fps(self):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time
        return int(fps)

def create_emotion_dataframe(history_list):
    """Detected emotions ki list ko Pandas DataFrame (CSV export ke liye) mein badalta hai."""
    if not history_list:
        return pd.DataFrame(columns=["Timestamp", "Emotion", "Confidence (%)"])
    return pd.DataFrame(history_list)

def format_log_entry(emotion, confidence):
    return {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Emotion": emotion.capitalize(),
        "Confidence (%)": confidence
    }
def load_asset(file_path):
    """Assets folder se HTML, CSS aur JS files ko read karta hai."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()