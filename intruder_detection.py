import cv2
import numpy as np
import face_recognition
import torch
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor

# Ensure the directory exists for storing photos
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# Load YOLOv5 Nano model (Fastest version) and enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', force_reload=True).to(device)
model.eval()
torch.set_grad_enabled(False)

# Load known faces
known_face_encodings = []
known_face_names = ["Family Member ", "Family Member ", "Family Member ", "Family Member"]
family_images = ["family_member_1.jpg", "family_member_2.jpg", "jk.jpg", "saleel.jpg"]

for i, file_name in enumerate(family_images):
    image_path = f"family_faces/{file_name}"
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        continue

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])  # Take first encoding
        print(f"Loaded face encoding for {known_face_names[i]}")
    else:
        print(f"Warning: No face found in {file_name}")

print(f"Total loaded known faces: {len(known_face_encodings)}")

# Set up webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0

# Telegram Bot Configuration
BOT_TOKEN = "7726687998:AAHHkM8x8bwo3PDsw7Of1jCHeSZGBORn0q8"
CHAT_ID = "812723861"

# Function to send an alert via Telegram
def send_telegram_alert(image_path, chat_id, token, message):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(image_path, "rb") as photo:
        payload = {"chat_id": chat_id, "caption": message}
        files = {"photo": photo}
        response = requests.post(url, data=payload, files=files)
        return response.json()

# Cooldown variables
alert_cooldown = 10  # Seconds before a new alert can be triggered
last_alert_time = 0

# Use Haar Cascade for face detection (fallback)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

executor = ThreadPoolExecutor(max_workers=2)  # Multi-threading

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    frame_resized = cv2.resize(frame, (640, 480))
    small_frame = cv2.resize(frame, (320, 240))  # Lower resolution for face recognition
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Run YOLO & Face Recognition in Parallel
    if frame_counter % 10 == 0:  # Process every 10th frame
        future_yolo = executor.submit(model, frame_resized)
    future_faces = executor.submit(face_recognition.face_encodings, rgb_frame, face_recognition.face_locations(rgb_frame))

    # YOLO Object Detection (Processed every 10th frame)
    if frame_counter % 10 == 0:
        results = future_yolo.result()
        boxes = results.xyxy[0].cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.5:
                label = model.names[int(cls)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # If a person is detected, send an alert
                if label == "person":
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        alert_message = "ALERT: Person detected!"
                        last_alert_time = current_time
                        filename = f"captured_images/person_{int(current_time)}.jpg"
                        cv2.imwrite(filename, frame)
                        response = send_telegram_alert(filename, CHAT_ID, BOT_TOKEN, alert_message)
                        print("Telegram Response:", response)

    # Use Haar Cascade for Face Detection (Fallback)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process Face Recognition
    face_encodings = future_faces.result()
    alert_message = ""

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Get face encoding for detected face
            face_encoding = face_encodings[0] if len(face_encodings) > 0 else None
            if face_encoding is None:
                continue

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                print(f"Matched: {name}")
            else:
                print("No match found.")

            # Alert if unknown
            if name == "Unknown":
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    alert_message = "ALERT: Unknown person detected!"
                    last_alert_time = current_time
                    filename = f"captured_images/unknown_{int(current_time)}.jpg"
                    cv2.imwrite(filename, frame)
                    response = send_telegram_alert(filename, CHAT_ID, BOT_TOKEN, alert_message)
                    print("Telegram Response:", response)

                color = (0, 0, 255)  # Red for unknown
            else:
                color = (255, 0, 0)  # Blue for recognized

            # Draw rectangle & name
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display alert message
    if alert_message:
        cv2.putText(frame, alert_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show output
    cv2.imshow("Intruder Detection System", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
