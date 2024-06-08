import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import logging
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the face detector and the trained model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the Keras model
classifier = load_model('model.h5')

# Freeze Batch Normalization layers (optional)
for layer in classifier.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Set page configuration
st.set_page_config(page_title="Streamlit Emotion Recognition App")
st.title("Emotion Recognition App")
st.caption("Powered by OpenCV, Streamlit")

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    logging.debug(f"Faces detected: {faces}")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        logging.debug("Face region extracted and resized")

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            try:
                prediction = classifier.predict(roi, verbose=0)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                logging.debug(f"Prediction: {label}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                logging.error(f"Prediction error: {e}")
        else:
            cv2.putText(image, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    processed_image = detect_emotion(image)
    st.image(processed_image, channels="BGR")
