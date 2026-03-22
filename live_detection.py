import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------
# Load your trained model
# -------------------------------
model = load_model("model.h5")

# -------------------------------
# Start webcam
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -------------------------------
# Real-time detection loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Optional: Flip camera
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to match model input
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img, verbose=0)[0][0]  # sigmoid output between 0 and 1

    # Interpret prediction
    if pred < 0.5:
        label = f"Biodegradable ✅ ({(1-pred)*100:.2f}%)"
        color = (0, 255, 0)  # green
    else:
        label = f"Non-Biodegradable ❌ ({pred*100:.2f}%)"
        color = (0, 0, 255)  # red

    # Display label
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("EcoBin Live Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()