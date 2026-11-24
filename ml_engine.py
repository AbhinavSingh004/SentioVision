# ml_engine.py
import cv2
from deepface import DeepFace
from config import DETECTOR_BACKEND, EMOTION_COLORS, DEFAULT_COLOR

def predict_emotion(frame):
    """
    Machine Learning / Deep Learning function:
    Frame leta hai -> Face detect karta hai -> Emotion predict karta hai -> Box draw karta hai.
    """
    detected_data = []

    try:
        # YAHAN MACHINE LEARNING / DEEP LEARNING MODEL CHAL RAHA HAI
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )

        # Agar ek image mein multiple faces hain toh loop chalega
        for face in results:
            # Face ke coordinates nikalna
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # ML Model dwara predict kiya gaya dominant emotion aur probabilities
            dominant_emotion = face['dominant_emotion']
            emotion_scores = face['emotion']
            confidence = emotion_scores[dominant_emotion]

            # Config file se emotion ke hisaab se color lena
            box_color = EMOTION_COLORS.get(dominant_emotion, DEFAULT_COLOR)

            # Face ke charo taraf rectangle aur text draw karna
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            label = f"{dominant_emotion.upper()} ({confidence:.1f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            detected_data.append({
                "emotion": dominant_emotion,
                "confidence": round(confidence, 2),
                "all_scores": emotion_scores
            })

    except Exception as e:
        pass

    return frame, detected_data