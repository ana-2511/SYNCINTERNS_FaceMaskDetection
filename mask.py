import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load pre-trained face mask detection model
model = load_model('D:\\anangsha\\Sync Intern\\face_mask_detection_model.h5')

# Load the face cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the path to the Kaggle dataset (update with the actual path)
dataset_path = 'E:\\Face Mask Dataset'

# Open the webcam (you may need to adjust the index based on your camera setup)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the face image for the model
        face_array = cv2.resize(face_roi, (224, 224))
        face_array = image.img_to_array(face_array)
        face_array = np.expand_dims(face_array, axis=0)

        # Make predictions using the model
        predictions = model.predict(face_array)
        mask_probability = predictions[0][0]

        # Display the result on the frame
        label = "Mask" if mask_probability > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(frame, f"{label}: {mask_probability:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Display the frame
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
