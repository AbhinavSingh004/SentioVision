# config.py

# DeepFace backend options: 'opencv', 'ssd', 'mtcnn', 'retinaface'
DETECTOR_BACKEND = "opencv"

# Har emotion ke liye alag BGR color (OpenCV BGR format use karta hai)
EMOTION_COLORS = {
    "happy": (0, 255, 0),       # Green
    "sad": (255, 0, 0),         # Blue
    "angry": (0, 0, 255),       # Red
    "surprise": (0, 255, 255),  # Yellow
    "fear": (255, 0, 255),      # Purple
    "disgust": (0, 128, 128),   # Olive
    "neutral": (200, 200, 200)  # Light Gray
}

DEFAULT_COLOR = (255, 255, 255)