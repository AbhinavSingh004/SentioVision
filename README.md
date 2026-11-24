# SentioVision — Real-Time Facial Emotion Recognition

SentioVision is a Deep Learning and Computer Vision application that detects human faces and predicts facial emotions in real time using pre-trained CNN models via DeepFace, TensorFlow/Keras, OpenCV, and Streamlit.

## Project Structure

```text
SentioVision/
├── assets/             # Custom HTML, CSS, and JavaScript UI components
│   ├── header.html
│   ├── style.css
│   └── script.js
├── app.py              # Main Streamlit web application
├── ml_engine.py        # DeepFace & CNN emotion detection engine
├── config.py           # Application settings and constants
├── utils.py            # Image processing and helper utilities
├── requirements.txt    # Python dependencies
└── .gitignore          # Ignored local/environment files
