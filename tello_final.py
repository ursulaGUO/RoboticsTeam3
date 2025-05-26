from djitellopy import Tello
import torch
import cv2
import time
import numpy as np
import torch
from pathlib import Path
import sys
import atexit  # ✅ NEW
import signal  # ✅ NEW

# Add YOLOv5 repo to path
YOLOV5_PATH = '/Users/blag/Documents/UChicago MS/2025 Spring/Robotics1/yolov5' 
sys.path.append(YOLOV5_PATH)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

import mediapipe as mp

# === Toggle Test Mode ===
TEST_MODE = False  # Set to False to enable real drone flight

# === Initialize Tello and Start Stream ===
tello = Tello()
tello.connect()
print(f"Current battery level: {tello.get_battery()}%")
tello.streamon()
frame_reader = tello.get_frame_read()


# === Safe Shutdown Setup ===
in_air = False  # Must be declared before safe_shutdown

def safe_shutdown():
    global in_air
    print("[INFO] Safe shutdown triggered...")
    if in_air:
        print("[ACTION] Landing drone")
        try:
            tello.land()
        except Exception as e:
            print(f"[ERROR] Failed to land: {e}")
        in_air = False
    try:
        tello.streamoff()
        tello.end()
    except Exception as e:
        print(f"[ERROR] Failed to close Tello: {e}")
    cv2.destroyAllWindows()

atexit.register(safe_shutdown)
signal.signal(signal.SIGINT, lambda sig, frame: exit(0))
signal.signal(signal.SIGTERM, lambda sig, frame: exit(0))

# === YOLOv5 Model Load ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = DetectMultiBackend(str(Path(YOLOV5_PATH) / 'yolov5s.pt'), device=device)
model.eval()
model.names = model.names  # COCO classes

# === MediaPipe Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def detect_gesture(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    return "thumbs_up" if thumb_tip < thumb_mcp else "thumbs_down" if thumb_tip > thumb_mcp else "none"

# === State ===
rotating = False
last_gesture_time = time.time()

# === Main Loop ===
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        frame = frame_reader.frame
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # === Gesture Detection ===
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = detect_gesture(hand_landmarks.landmark)
                print(f"[INFO] Gesture detected: {gesture}")

                current_time = time.time()
                if current_time - last_gesture_time > 3:
                    if gesture == "thumbs_up" and not in_air:
                        print("[ACTION] Takeoff triggered")
                        if not TEST_MODE:
                            tello.takeoff()
                        in_air = True
                        last_gesture_time = current_time
                    elif gesture == "thumbs_down" and in_air:
                        print("[ACTION] Landing triggered")
                        if not TEST_MODE:
                            tello.land()
                        in_air = False
                        last_gesture_time = current_time

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === YOLO Object Detection ===
        img = letterbox(frame, new_shape=640)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW and BGR to RGB
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor, augment=False)
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)

        cup_found = False
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = model.names[int(cls)]
                    if label == "cup":
                        cup_found = True
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # === Drone Rotation and Move Toward Cup Logic ===
        if in_air:
            if not cup_found:
                if not rotating:
                    print("[INFO] Cup not found, rotating...")
                    rotating = True
                    rotation_step = 20
                    total_rotation = 0
                else:
                    if total_rotation < 360:
                        print(f"[INFO] Rotating {rotation_step} degrees...")
                        if not TEST_MODE:
                            tello.rotate_clockwise(rotation_step)
                        total_rotation += rotation_step
                        time.sleep(2)  # Let the frame update
                    else:
                        print("[INFO] Full rotation done. Cup not found.")
                        rotating = False
                        total_rotation = 0
            else:
                if rotating:
                    print("[INFO] Cup found, stopping rotation.")
                    rotating = False
                    total_rotation = 0

                # === Move Forward Toward Cup with Quadrant Logic ===
                print("[INFO] Evaluating object position and size before moving forward...")
                if not TEST_MODE:
                    box_width = int(xyxy[2]) - int(xyxy[0])
                    box_center_x = int((xyxy[0] + xyxy[2]) / 2)
                    box_center_y = int((xyxy[1] + xyxy[3]) / 2)
                    frame_h, frame_w = frame.shape[:2]

                    # Define middle 25% region
                    mid_x1 = int(frame_w * 0.375)
                    mid_x2 = int(frame_w * 0.625)
                    mid_y1 = int(frame_h * 0.375)
                    mid_y2 = int(frame_h * 0.625)

                    if mid_x1 <= box_center_x <= mid_x2 and mid_y1 <= box_center_y <= mid_y2:
                        # Center region: move forward if not close enough
                        if box_width < 0.4 * frame_w:
                            print("[INFO] Cup in center, moving forward")
                            tello.move_forward(20)
                            time.sleep(3)
                        else:
                            print(f"[INFO] Cup is close enough (width = {box_width}px), not moving forward.")
                    else:
                        # Determine quadrant and move accordingly
                        if box_center_x < frame_w // 2 and box_center_y < frame_h // 2:
                            print("[INFO] Cup in upper left quadrant: moving up, left, and forward")
                            tello.move_up(20)
                            tello.move_left(20)
                            tello.move_forward(20)
                        elif box_center_x >= frame_w // 2 and box_center_y < frame_h // 2:
                            print("[INFO] Cup in upper right quadrant: moving up, right, and forward")
                            tello.move_up(20)
                            tello.move_right(20)
                            tello.move_forward(20)
                        elif box_center_x < frame_w // 2 and box_center_y >= frame_h // 2:
                            print("[INFO] Cup in lower left quadrant: moving down, left, and forward")
                            tello.move_down(20)
                            tello.move_left(20)
                            tello.move_forward(20)
                        elif box_center_x >= frame_w // 2 and box_center_y >= frame_h // 2:
                            print("[INFO] Cup in lower right quadrant: moving down, right, and forward")
                            tello.move_down(20)
                            tello.move_right(20)
                            tello.move_forward(20)
                        time.sleep(3)


        # === Show Feed ===
        title = "Tello YOLO Stream (Test Mode)" if TEST_MODE else "Tello YOLO Stream"
        cv2.imshow(title, frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

# === Cleanup handled by safe_shutdown ===
