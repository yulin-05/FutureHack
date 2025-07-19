import cv2
import numpy as np
import mediapipe as mp
import os
import time
import json

gesture_label = "thank you"     #name of gesture to be collected
category_folder = "terms"       #category folder
record_seconds = 3              #duration
fps = 30
sequence_length = record_seconds * fps

save_folder = os.path.join("dataset", category_folder, gesture_label)
os.makedirs(save_folder, exist_ok=True)

#mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

#camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, fps)

sample_count = 0
recording = False
start_time = None
motion_buffer = []

print(f"✨ Ready to collect gesture: {gesture_label}")
print("🎬 Press 's' to start recording")
print("❌ Press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #Overlay info
    cv2.putText(bgr, f"Gesture: {gesture_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(bgr, f"Samples Collected: {sample_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    if recording:
        elapsed = time.time() - start_time
        cv2.putText(bgr, f"Recording: {elapsed:.1f}s", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                if len(landmarks) == 63:
                    motion_buffer.append(landmarks)

        if elapsed >= record_seconds:
            if len(motion_buffer) >= 0:
                timestamp = int(time.time() * 1000)
                filename = f"{gesture_label}_{timestamp}.json"
                filepath = os.path.join(save_folder, filename)
                with open(filepath, 'w') as f:
                    json.dump(motion_buffer[:sequence_length], f)
                print(f"✅ Saved: {filename}")
                sample_count += 1
            else:
                print("⚠ Not enough data collected. Try again.")
            motion_buffer = []
            recording = False

    cv2.imshow("Motion Gesture Capture", bgr)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        print("⏺ Recording started")
        recording = True
        start_time = time.time()
        motion_buffer = []
    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()