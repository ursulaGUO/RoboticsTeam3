from djitellopy import Tello
import torch
import cv2
import time
import numpy as np
from pathlib import Path
import sys
import atexit
import signal
import threading
import queue
import sounddevice as sd
import mediapipe as mp
from vosk import Model, KaldiRecognizer
import json

# === YOLOv5 Setup ===
# Note: we had to append path to yolov5 file location first
# before importing the following functions
# So we could not follow Pep8 strictly about doing all the imports at the top
YOLOV5_PATH = "/Users/blag/Documents/UChicago MS/2025 Spring/Robotics1/yolov5"
sys.path.append(YOLOV5_PATH)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


# === MediaPipe Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# === Helper ===
def detect_gesture(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    return (
        "thumbs_up"
        if thumb_tip < thumb_mcp
        else "thumbs_down"
        if thumb_tip > thumb_mcp
        else "none"
    )


# === Shared State ===
in_air = [False]  # Mutable container for cross-thread reference
gesture_queue = queue.Queue()


# === Gesture Thread Function ===
def gesture_listener(frame_reader, in_air_ref, gesture_queue):
    last_gesture_time = time.time()
    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            frame = frame_reader.frame
            if frame is None:
                continue
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = detect_gesture(hand_landmarks.landmark)
                    print(f"[GESTURE] Detected: {gesture}")
                    current_time = time.time()
                    if current_time - last_gesture_time > 3:
                        if gesture == "thumbs_up" and not in_air_ref[0]:
                            gesture_queue.put("fly")
                            last_gesture_time = current_time
                        elif gesture == "thumbs_down" and in_air_ref[0]:
                            gesture_queue.put("land")
                            last_gesture_time = current_time
            time.sleep(0.1)


# === Vosk Model Setup ===
VOSK_MODEL_PATH = "/Users/blag/Documents/UChicago MS/2025 Spring/Robotics1/vosk-model-small-en-us-0.15"
print("[INFO] Loading Vosk model...")
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
audio_queue = queue.Queue()
command_queue = queue.Queue()


# === Vosk Audio Callback ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[WARNING] {status}")
    audio_queue.put(bytes(indata))


# === Voice Command Listener Thread ===
def voice_command_listener():
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        print("🎤 Say 'fly' to take off or 'land' to land the drone.")
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                if text:
                    print(f"[YOU SAID] {text}")
                    if any(cmd in text for cmd in ["fly", "land", "exit", "quit"]):
                        command_queue.put(text)


# === Initialize Tello and Stream ===
TEST_MODE = False
tello = Tello()
tello.connect()
print(f"[INFO] Battery level: {tello.get_battery()}%")
tello.streamon()
frame_reader = tello.get_frame_read()


# === Safe Shutdown ===
def safe_shutdown():
    print("[INFO] Safe shutdown triggered...")
    if in_air[0]:
        print("[ACTION] Landing drone")
        try:
            tello.land()
        except Exception as e:
            print(f"[ERROR] Failed to land: {e}")
        in_air[0] = False
    try:
        tello.streamoff()
        tello.end()
    except Exception as e:
        print(f"[ERROR] Failed to close Tello: {e}")
    cv2.destroyAllWindows()


atexit.register(safe_shutdown)
signal.signal(signal.SIGINT, lambda sig, frame: exit(0))
signal.signal(signal.SIGTERM, lambda sig, frame: exit(0))

# === Start Gesture and Voice Thread ===
gesture_thread = threading.Thread(
    target=gesture_listener, args=(frame_reader, in_air, gesture_queue), daemon=True
)
gesture_thread.start()

voice_thread = threading.Thread(target=voice_command_listener, daemon=True)
voice_thread.start()

# === YOLO Model Load ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = DetectMultiBackend(str(Path(YOLOV5_PATH) / "yolov5s.pt"), device=device)
model.eval()
model.names = model.names

# === Main Loop ===
rotating = False
total_rotation = 0

while True:
    # === Handle Gesture Commands ===
    if not gesture_queue.empty():
        command = gesture_queue.get()
        if command == "fly" and not in_air[0]:
            print("[ACTION] Gesture Takeoff")
            if not TEST_MODE:
                tello.takeoff()
            in_air[0] = True
        elif command == "land" and in_air[0]:
            print("[ACTION] Gesture Landing")
            if not TEST_MODE:
                tello.land()
            in_air[0] = False

    # === Read and Process Frame ===
    frame = frame_reader.frame
    if frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))
    img = letterbox(frame, new_shape=640)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)

    cup_found = False
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(
                img_tensor.shape[2:], det[:, :4], frame.shape
            ).round()
            for *xyxy, conf, cls in det:
                label = model.names[int(cls)]
                if label == "cup":
                    cup_found = True
                    cv2.rectangle(
                        frame,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

    # === Move Toward Cup ===
    if in_air[0]:
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
                    time.sleep(2)
                else:
                    print("[INFO] Full rotation done. Cup not found.")
                    rotating = False
                    total_rotation = 0
        else:
            if rotating:
                print("[INFO] Cup found, stopping rotation.")
                rotating = False
                total_rotation = 0

            if not TEST_MODE:
                box_width = int(xyxy[2]) - int(xyxy[0])
                box_center_x = int((xyxy[0] + xyxy[2]) / 2)
                box_center_y = int((xyxy[1] + xyxy[3]) / 2)
                frame_h, frame_w = frame.shape[:2]

                mid_x1 = int(frame_w * 0.375)
                mid_x2 = int(frame_w * 0.625)
                mid_y1 = int(frame_h * 0.375)
                mid_y2 = int(frame_h * 0.625)

                if (
                    mid_x1 <= box_center_x <= mid_x2
                    and mid_y1 <= box_center_y <= mid_y2
                ):
                    if box_width < 0.4 * frame_w:
                        print("[INFO] Cup in center, moving forward")
                        tello.move_forward(20)
                        time.sleep(3)
                else:
                    if box_center_x < frame_w // 2 and box_center_y < frame_h // 2:
                        print("[INFO] Cup in upper left quadrant")
                        tello.move_up(20)
                        tello.move_left(20)
                        tello.move_forward(20)
                    elif box_center_x >= frame_w // 2 and box_center_y < frame_h // 2:
                        print("[INFO] Cup in upper right quadrant")
                        tello.move_up(20)
                        tello.move_right(20)
                        tello.move_forward(20)
                    elif box_center_x < frame_w // 2 and box_center_y >= frame_h // 2:
                        print("[INFO] Cup in lower left quadrant")
                        tello.move_down(20)
                        tello.move_left(20)
                        tello.move_forward(20)
                    elif box_center_x >= frame_w // 2 and box_center_y >= frame_h // 2:
                        print("[INFO] Cup in lower right quadrant")
                        tello.move_down(20)
                        tello.move_right(20)
                        tello.move_forward(20)
                    time.sleep(3)

    # === Show Feed ===
    title = "Tello YOLO Stream (Test Mode)" if TEST_MODE else "Tello YOLO Stream"
    cv2.imshow(title, frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# === Final Cleanup ===
safe_shutdown()
