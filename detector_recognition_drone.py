import json
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello
from deepface import DeepFace

# ======================
# 0. Configuration
# ======================
REFERENCE_IMAGE = "final_project/Ursula_pic.png" 
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.3  # smaller = stricter match
CONF_THRESHOLD = 0.4
IMG_SIZE = 640

# ======================
# 1. Initialize models
# ======================
print("Computing embedding for reference face...")
ref_embedding = DeepFace.represent(
    img_path=REFERENCE_IMAGE,
    model_name="Facenet512",
    enforce_detection=True
)[0]["embedding"]
ref_embedding = np.array(ref_embedding)
print("Reference embedding loaded.")

# Human detector (for person bounding boxes)
person_model = YOLO("yolo11n.pt")  # can use "yolo11s.pt" for more accuracy

# Pose model (for keypoints)
pose_model = YOLO("yolo11n-pose.pt")

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


# ======================
# 2. Connect to Tello
# ======================
tello = Tello()
print("Connecting to drone...")
tello.connect()
print(f"Connected. Battery: {tello.get_battery()}%")

tello.streamon()
time.sleep(2)
frame_reader = tello.get_frame_read()

print("Starting live detection... Press 'q' to quit.")

# ======================
# 3. Helper functions
# ======================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_hand_raised(keypoints):
    """
    Given 17 COCO keypoints, return True if either hand is above the head.
    keypoints: np.ndarray of shape (17, 3)
    """
    if keypoints is None or len(keypoints) < 11:
        return False
    y = keypoints[:, 1]  # y-coordinates
    head_y = np.min(y[:5])  # top of head area (nose/eyes/ears)
    left_wrist_y, right_wrist_y = y[9], y[10]
    left_shoulder_y, right_shoulder_y = y[5], y[6]
    left_up = left_wrist_y < min(left_shoulder_y, head_y)
    right_up = right_wrist_y < min(right_shoulder_y, head_y)
    return left_up or right_up

def match_pose_to_person(person_box, pose_keypoints):
    """
    Check if the pose keypoints fall mostly inside the person box.
    Returns True if they correspond to the same person.
    """
    x1, y1, x2, y2 = person_box
    if pose_keypoints is None:
        return False
    kps = pose_keypoints[:, :2]
    inside = np.logical_and.reduce([
        kps[:, 0] > x1,
        kps[:, 0] < x2,
        kps[:, 1] > y1,
        kps[:, 1] < y2
    ])
    return np.mean(inside) > 0.6  # >60% keypoints inside box

# ======================
# 4. Live Loop
# ======================
while True:
    frame = frame_reader.frame
    if frame is None:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_frame = frame.copy()
    detections = []

    # ---- 4.1 Run Human Detection ----
    person_results = person_model.predict(
        source=frame_rgb,
        classes=[0],  # person class
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False
    )

    # ---- 4.2 Run Pose Detection ----
    pose_results = pose_model.predict(
        source=frame_rgb,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False
    )

    # Collect all pose keypoints
    all_keypoints = []
    for r in pose_results:
        if r.keypoints is not None:
            for kp in r.keypoints:
                kps = kp.data[0].cpu().numpy()
                all_keypoints.append(kps)

    # ---- 4.3 Process each detected person ----
    for det in person_results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        crop = frame_rgb[y1:y2, x1:x2]
        label = "Unknown"
        posture = "Normal"

        # ---- Face recognition using person box ----
        try:
            if crop.size != 0:
                detected = DeepFace.represent(
                    img_path=crop,
                    model_name="Facenet512",
                    enforce_detection=False
                )
                if detected:
                    det_embedding = np.array(detected[0]["embedding"])
                    sim = cosine_similarity(ref_embedding, det_embedding)
                    if sim > (1 - THRESHOLD):
                        label = "Ursula"
        except Exception as e:
            print(f"DeepFace error: {e}")

        # ---- Posture recognition by matching pose ----
        matched_pose = None
        for kps in all_keypoints:
            if match_pose_to_person((x1, y1, x2, y2), kps):
                matched_pose = kps
                break
        if matched_pose is not None and is_hand_raised(matched_pose):
            posture = "Hand Raised"

        # ---- Draw only the person box ----
        color = (0, 255, 0) if posture == "Hand Raised" else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{label} | {posture}",
                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detections.append({
            "label": label,
            "posture": posture,
            "box": [x1, y1, x2, y2]
        })

    # ---- Overlay info ----
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    info_text = f"Time: {timestamp} | Detections: {len(detections)}"
    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # ---- Display ----
    cv2.imshow("Tello Face + Posture Recognition", annotated_frame)

    # ---- Save JSON log ----
    record = {"timestamp": timestamp, "detections": detections}
    with open("pose_detections.json", "w") as f:
        json.dump(record, f, indent=2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================
# 5. Cleanup
# ======================
print("Stopping stream and closing window...")
tello.streamoff()
cv2.destroyAllWindows()
print("All done.")
