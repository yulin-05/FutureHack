import cv2
import csv
import mediapipe as mp
import os
import time
import pandas as pd

#gestures to be collected
GESTURE_LABELS = {
    #numbers
    ord('1'): '1', ord('2'): '2', ord('3'): '3', ord('4'): '4', ord('5'): '5', ord('6'): '6', ord('7'): '7', ord('8'): '8', ord('9'): '9',
    #alphabets(static)
    ord('a'): 'A', ord('b'): 'B', ord('c'): 'C', ord('d'): 'D', ord('e'): 'E', ord('f'): 'F', ord('g'): 'G', ord('h'): 'H', ord('i'): 'I',
    ord('k'): 'K', ord('l'): 'L', ord('m'): 'M', ord('n'): 'N', ord('o'): 'O', ord('p'): 'P', ord('q'): 'Q', ord('r'): 'R', ord('s'): 'S',
    ord('t'): 'T', ord('u'): 'U', ord('v'): 'V', ord('w'): 'W', ord('x'): 'X', ord('y'): 'Y'
}
SAVE_FILENAME = "gesture_data.csv"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_DELAY = 0.05

#create csv file if not exist
file_exists = os.path.isfile(SAVE_FILENAME)
if not file_exists:
    headers = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x','y','z']]
    with open(SAVE_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

#open csv file to append
csv_file = open(SAVE_FILENAME, mode='a', newline='')
csv_writer = csv.writer(csv_file)

#mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

#camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("📷 Press number keys 1-9, 'a-y' (except 'j' and 'z')")
print("🗑 To delete data, press 'd' and input label to delete")

prev_time = time.time()

while cap.isOpened():
    current_time = time.time()
    if current_time - prev_time < FPS_DELAY:
        continue
    prev_time = current_time

    success, frame = cap.read()
    if not success:
        print("❌ Camera read failed!")
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if key in GESTURE_LABELS:
                label = GESTURE_LABELS[key]
                csv_writer.writerow([label] + landmarks)
                csv_file.flush()
                print(f"💾 Saved: {label}")

    if key == ord('0'):
        print("👋 Bye bye! Data collection ended.")
        break

    cv2.imshow("🖐 Gesture Collector", image)

csv_file.close()
cap.release()
cv2.destroyAllWindows()