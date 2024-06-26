import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import logging

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
st.set_page_config(page_title="Streamlit WebCam App")
st.title("Webcam Display Streamlit App")
st.caption("Powered by OpenCV, Streamlit")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible. Please check the connection and permissions.")
        logging.debug("Webcam not accessible")
        return

    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            logging.debug("Failed to capture image")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logging.debug("Converted frame to grayscale")

        # Detect faces in the grayscale frame
        faces = face_classifier.detectMultiScale(gray)
        logging.debug(f"Faces detected: {faces}")

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
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
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    logging.debug(f"Prediction: {label}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    logging.error(f"Prediction error: {e}")
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame in the Streamlit app
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        logging.debug("Frame displayed")

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    cap.release()
    logging.debug("Webcam released")
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
