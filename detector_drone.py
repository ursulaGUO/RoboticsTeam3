import json
import cv2
import torch
import time
from ultralytics import YOLO
from djitellopy import Tello

# ======================
# 1. Initialize YOLO model
# ======================
model = YOLO("yolo11n.pt")

# ======================
# 2. Connect to Tello drone
# ======================
tello = Tello()
print("Connecting to drone...")
tello.connect()
print(f"Connected. Battery: {tello.get_battery()}%")

# Start video stream
tello.streamon()
time.sleep(2)
frame_read = tello.get_frame_read()

print("Starting live detection... Press 'q' in the window to quit.")

# ======================
# 3. Live detection loop
# ======================
while True:
    frame = frame_read.frame
    if frame is None:
        continue

    # Run YOLO prediction
    results = model.predict(
        source=frame,
        conf=0.25,        # confidence threshold
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    # Parse detections
    humans = []
    for box in results[0].boxes:
        cls = int(box.cls)
        if results[0].names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            humans.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Human", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw overlay text
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    human_count = len(humans)
    info_text = f"Time: {timestamp} | Humans: {human_count}"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show live frame with overlay
    cv2.imshow("YOLOv11 + Tello Drone", frame)

    # Log to file
    record = {"timestamp": timestamp, "count": human_count, "humans": humans}
    with open("detections.json", "w") as f:
        json.dump(record, f, indent=2)

    # Break condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================
# 4. Cleanup
# ======================
print("Stopping stream and closing window...")
tello.streamoff()
cv2.destroyAllWindows()
print("All done. Detections saved to detections.json")
