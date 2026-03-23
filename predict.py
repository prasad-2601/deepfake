import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load model once
model = load_model("models/deepfake_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return "Invalid Image"

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "FAKE"
    else:
        return "REAL"