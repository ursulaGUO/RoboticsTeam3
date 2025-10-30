import json
import cv2
import torch
import time
from ultralytics import YOLO
from djitellopy import Tello
from deepface import DeepFace
import numpy as np

# ======================
# 0. Configuration
# ======================
# Path to your reference face image (user-uploaded)
REFERENCE_IMAGE = "final_project/Ursula_pic.png"  # Replace with uploaded file name

# Choose distance metric and threshold
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.3  # smaller = stricter; tune as needed

# ======================
# 1. Initialize YOLO model
# ======================
model = YOLO("yolo11n.pt")

# ======================
# 2. Load reference face embedding
# ======================
print("Computing embedding for reference face...")
ref_embedding = DeepFace.represent(
    img_path=REFERENCE_IMAGE,
    model_name="Facenet512",   # can also use ArcFace, VGG-Face, etc.
    enforce_detection=True
)[0]["embedding"]
ref_embedding = np.array(ref_embedding)

print("Reference embedding loaded.")

# ======================
# 3. Connect to Tello drone
# ======================
tello = Tello()
print("Connecting to drone...")
tello.connect()
print(f"Connected. Battery: {tello.get_battery()}%")

tello.streamon()
time.sleep(2)
frame_read = tello.get_frame_read()


print("Starting live detection... Press 'q' in the window to quit.")

# ======================
# 4. Live detection + recognition loop
# ======================
while True:
    frame = frame_read.frame
    if frame is None:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO human detection
    results = model.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    humans = []
    for box in results[0].boxes:
        cls = int(box.cls)
        if results[0].names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            humans.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

            # Crop face region from the person box
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            try:
                # Compute embedding for detected face
                detected = DeepFace.represent(
                    img_path=person_crop,
                    model_name="Facenet512",
                    enforce_detection=False
                )
                if detected:
                    det_embedding = np.array(detected[0]["embedding"])
                    # Cosine similarity
                    similarity = np.dot(ref_embedding, det_embedding) / (
                        np.linalg.norm(ref_embedding) * np.linalg.norm(det_embedding)
                    )
                    label = "Ursula" if similarity > (1 - THRESHOLD) else "Unknown"
                else:
                    label = "Unknown"
            except Exception as e:
                label = "Error"
                print(f"DeepFace error: {e}")

            # Draw bounding box and label
            color = (0, 255, 0) if label == "Ursula" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Overlay info
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    info_text = f"Time: {timestamp} | Humans: {len(humans)}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("YOLOv11 + DeepFace Recognition", frame)

    # Save detections if desired
    record = {"timestamp": timestamp, "count": len(humans), "humans": humans}
    with open("detections.json", "w") as f:
        json.dump(record, f, indent=2)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ======================
# 5. Cleanup
# ======================
print("Stopping stream and closing window...")
tello.streamoff()
cv2.destroyAllWindows()
print("All done.")
