import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = tf.keras.models.load_model(r"model\emotion5.h5")

# Emotion labels
ALL_EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']  # ✅ 5 classes
TARGET_EMOTIONS = ['Happy', 'Sad', 'Neutral']

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("❌ Error loading Haar Cascade file.")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not access webcam.")
    exit()

print("✅ Live Emotion Detection Started! (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame.")
        break

    # Convert to gray for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Extract face in color (not grayscale)
        face = frame[y:y + h, x:x + w]

        # Preprocess for model input
        face_resized = cv2.resize(face, (224, 224))             # ✅ Match training size
        face_array = face_resized.astype("float32") / 255.0
        face_array = np.expand_dims(face_array, axis=0)         # (1, 224, 224, 3)

        # Predict emotion
        preds = model.predict(face_array, verbose=0)[0]
        emotion_label = ALL_EMOTIONS[np.argmax(preds)]

        # Only show Happy, Sad, or Neutral
        if emotion_label not in TARGET_EMOTIONS:
            emotion_label = "Neutral"

        # Assign color
        color = (0, 255, 0) if emotion_label == "Happy" else \
                (0, 0, 255) if emotion_label == "Sad" else \
                (255, 255, 0)

        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Live Emotion Detection (Happy, Sad, Neutral)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
