import json
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello
from deepface import DeepFace

# ======================import json
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

# Use Metal if available
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
    """Return True if either hand is above the head."""
    if keypoints is None or len(keypoints) < 11:
        return False
    y = keypoints[:, 1]
    head_y = np.min(y[:5])
    left_wrist_y, right_wrist_y = y[9], y[10]
    left_shoulder_y, right_shoulder_y = y[5], y[6]
    left_up = left_wrist_y < min(left_shoulder_y, head_y)
    right_up = right_wrist_y < min(right_shoulder_y, head_y)
    return left_up or right_up

def match_pose_to_person(person_box, pose_keypoints):
    """Check if the pose keypoints fall mostly inside the person box."""
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

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_frame = frame.copy()
    detections = []

    # ---- 4.1 Run Human Detection ----
    person_results = person_model.predict(
        source=frame,
        classes=[0],  # person class
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False
    )

    # ---- 4.2 Run Pose Detection ----
    pose_results = pose_model.predict(
        source=frame,
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
    num_humans = 0
    num_hand_raising = 0
    poi_detected = False  # “Person of Interest” flag

    for det in person_results[0].boxes:
        num_humans += 1
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        crop = frame[y1:y2, x1:x2]
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
                        poi_detected = True
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
            num_hand_raising += 1

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
    info_text1 = f"Time: {timestamp}"
    info_text2 = f"Number of Humans: {num_humans}"
    info_text3 = f"Detected POI: {poi_detected}"
    info_text4 = f"Number of Hand Raising: {num_hand_raising}"

    # Draw background rectangle for UI
    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 100), (0, 0, 0), -1)

    # Display all info lines
    cv2.putText(annotated_frame, info_text1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, info_text2, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, info_text3, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if poi_detected else (0, 0, 255), 2)
    cv2.putText(annotated_frame, info_text4, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    # ---- Display ----
    cv2.imshow("Tello Face + Posture Recognition", annotated_frame)

    # ---- Save JSON log ----
    record = {
        "timestamp": timestamp,
        "num_humans": num_humans,
        "num_hand_raising": num_hand_raising,
        "poi_detected": poi_detected,
        "detections": detections
    }
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

# Use Metal if available
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
    """Return True if either hand is above the head."""
    if keypoints is None or len(keypoints) < 11:
        return False
    y = keypoints[:, 1]
    head_y = np.min(y[:5])
    left_wrist_y, right_wrist_y = y[9], y[10]
    left_shoulder_y, right_shoulder_y = y[5], y[6]
    left_up = left_wrist_y < min(left_shoulder_y, head_y)
    right_up = right_wrist_y < min(right_shoulder_y, head_y)
    return left_up or right_up

def match_pose_to_person(person_box, pose_keypoints):
    """Check if the pose keypoints fall mostly inside the person box."""
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

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_frame = frame.copy()
    detections = []

    # ---- 4.1 Run Human Detection ----
    person_results = person_model.predict(
        source=frame,
        classes=[0],  # person class
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False
    )

    # ---- 4.2 Run Pose Detection ----
    pose_results = pose_model.predict(
        source=frame,
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
    num_humans = 0
    num_hand_raising = 0
    poi_detected = False  # “Person of Interest” flag

    for det in person_results[0].boxes:
        num_humans += 1
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        crop = frame[y1:y2, x1:x2]
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
                        poi_detected = True
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
            num_hand_raising += 1

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
    info_text1 = f"Time: {timestamp}"
    info_text2 = f"Number of Humans: {num_humans}"
    info_text3 = f"Detected POI: {poi_detected}"
    info_text4 = f"Number of Hand Raising: {num_hand_raising}"

    # Draw background rectangle for UI
    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 100), (0, 0, 0), -1)

    # Display all info lines
    cv2.putText(annotated_frame, info_text1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, info_text2, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated_frame, info_text3, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if poi_detected else (0, 0, 255), 2)
    cv2.putText(annotated_frame, info_text4, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    # ---- Display ----
    cv2.imshow("Tello Face + Posture Recognition", annotated_frame)

    # ---- Save JSON log ----
    record = {
        "timestamp": timestamp,
        "num_humans": num_humans,
        "num_hand_raising": num_hand_raising,
        "poi_detected": poi_detected,
        "detections": detections
    }
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
