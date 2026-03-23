import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/deepfake_model.h5")

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fake_count = 0
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        prediction = model.predict(frame)[0][0]

        if prediction > 0.5:
            fake_count += 1

        total += 1

    cap.release()

    if total == 0:
        return "Invalid Video"

    if fake_count > total / 2:
        return "FAKE VIDEO"
    else:
        return "REAL VIDEO"