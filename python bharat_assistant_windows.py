# ==============================================================
# 🧠 Bharat Assistant: AI + Voice + Vision (Windows Edition)
# ==============================================================

import cv2
import numpy as np
import torch
import time
import os
import platform
from ultralytics import YOLO
from gtts import gTTS
import subprocess

# ==============================================================
# 1️⃣ Text-to-Speech (Windows Compatible)
# ==============================================================

def speak(text, save_audio=False):
    print(f"🔊 Speaking: {text}")
    tts = gTTS(text=text, lang='en', slow=False)
    audio_path = "object_detected_audio.mp3"
    tts.save(audio_path)
    
    if not save_audio:
        play_audio(audio_path)
    
    return audio_path


def play_audio(audio_path):
    system = platform.system()
    if system == "Windows":
        # Use Windows built-in player
        subprocess.run(["start", "", audio_path], shell=True)
    elif system == "Darwin":
        os.system(f"afplay {audio_path}")
    else:
        os.system(f"mpg123 {audio_path}")

# ==============================================================
# 2️⃣ YOLO Model Initialization
# ==============================================================

# Use YOLOv8 Small (faster and good accuracy)
model = YOLO("yolov8s.pt")

# ==============================================================
# 3️⃣ Direction Logic
# ==============================================================

def suggest_direction(detections, frame_width):
    left_count = 0
    right_count = 0

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]
        center_x = (x1 + x2) / 2
        if center_x < frame_width / 2:
            left_count += 1
        else:
            right_count += 1

    if left_count == 0 and right_count == 0:
        return "Path is clear. You can move forward."
    elif left_count > right_count:
        return "Obstacle on left. Move slightly right."
    elif right_count > left_count:
        return "Obstacle on right. Move slightly left."
    else:
        return "Objects on both sides. Proceed cautiously."

# ==============================================================
# 4️⃣ Live YOLO + Voice Detection (Windows Camera)
# ==============================================================

def live_yolo_with_voice(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    time.sleep(1.5)  # Allow camera to initialize

    if not cap.isOpened():
        print("❌ Could not open Windows camera.")
        return
    
    print("✅ Live detection started. Press 'q' to stop.")
    last_direction = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Frame not received. Retrying...")
                continue

            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, verbose=False)
            annotated = results[0].plot()

            frame_width = frame.shape[1]
            direction = suggest_direction(results[0].boxes, frame_width)

            cv2.putText(annotated, direction, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("🧠 Bharat Assistant - YOLO Live (Windows)", annotated)

            # Speak only when direction changes
            if direction != last_direction:
                speak(direction)
                last_direction = direction

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 'Q' pressed. Stopping detection...")
                break
    except KeyboardInterrupt:
        print("\n🧠 Interrupted manually.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Detection stopped safely.")

# ==============================================================
# 5️⃣ Run the Assistant
# ==============================================================

if __name__ == "__main__":
    speak("System initialized successfully. Windows live detection is ready.")
    live_yolo_with_voice(0)
