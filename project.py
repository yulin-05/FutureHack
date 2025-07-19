import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import pyttsx3
import threading
import time
import pyperclip
import os 
from playsound import playsound
from collections import Counter

SEQUENCE_LENGTH = 70

if "smodel" not in st.session_state:
    st.session_state.smodel = tf.keras.models.load_model("static_gesture_model.keras", compile=False)
    st.session_state.mmodel = tf.keras.models.load_model("motion_gesture_model.keras", compile=False)
    st.session_state.s_encoder = joblib.load("static_encoder.pkl") ####################
    st.session_state.m_encoder = joblib.load("motion_label_encoder.pkl")

smodel = st.session_state.smodel
mmodel = st.session_state.mmodel
s_encoder = st.session_state.s_encoder
m_encoder = st.session_state.m_encoder


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
speak_lock = threading.Lock()

def speak(text):
    with speak_lock:
        engine.say(text)
        engine.runAndWait()

def play_mp3_for_gesture(gesture):

    gesture_map = {
        "0": "ZERO.mp3",
        "1": "ONE.mp3",
        "2": "TWO.mp3",
        "3": "THREE.mp3",
        "4": "FOUR.mp3",
        "5": "FIVE.mp3",
        "6": "SIX.mp3",
        "7": "SEVEN.mp3",
        "8": "EIGHT.mp3",
        "9": "NINE.mp3",
        "LOVE": "LOVE.mp3",
        "OK": "OK.mp3",
        "A": "A.mp3",
        "B": "B.mp3",
        "C": "C.mp3",
        "D": "D.mp3",
        "E": "E.mp3",
        "F": "F.mp3",
        "G": "G.mp3",
        "H": "H.mp3",
        "I": "I.mp3",
        "j": "J.mp3",
        "K": "K.mp3",
        "L": "L.mp3",
        "M": "M.mp3",
        "N": "N.mp3",
        "O": "O.mp3",
        "P": "P.mp3",
        "Q": "Q.mp3",
        "R": "R.mp3",
        "S": "S.mp3",
        "T": "T.mp3",
        "U": "U.mp3",
        "V": "V.mp3",
        "W": "W.mp3",
        "X": "X.mp3",
        "Y": "Y.mp3",
        "z": "Z.mp3",
        "hello": "HELLO.mp3",
        "fine": "FINE.mp3",
        "how are you": "HOWAREYOU.mp3",
        "i love you": "ILOVEYOU.mp3",
        "my": "MY.mp3",
        "name": "NAME.mp3",
        "no": "NO.mp3",
        "please": "PLEASE.mp3",
        "sorry": "SORRY.mp3",
        "thank you": "THANKYOU.mp3",
        "what": "WHAT.mp3",
        "yes": "YES.mp3",
        "you": "YOU.mp3",
        "me": "ME.mp3",
        "I": "I.mp3"
    }

    mp3_file = gesture_map.get(gesture)

    if mp3_file:

        mp3_path = os.path.join(os.getcwd(), mp3_file)

        if os.path.exists(mp3_path):

            threading.Thread(target=playsound, args=(mp3_path,), daemon=True).start()

st.set_page_config(page_title="Gesture Translator", layout="wide")
st.markdown("""
    <style>
    body { background-color: #111 !important; color: white; }
    .stButton>button {
        background-color: #222; color: white;
        border: 1px solid #444; border-radius: 6px;
        padding: 6px 14px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00ffcc;
        color: black;
        font-weight: bold;
    }
    .control-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def init_state():
    defaults = {
        "history": [],
        "motion_buffer": [],
        "last_spoken_label": "",
        "last_spoken_time": 0,
        "cooldown": 2,
        "camera_on": False,
        "end_call": False,
        "copy_success": False,
        "gesture_timer": 0,
        "gesture_candidate": "",
        "stable_gesture": "",
        "detect_on": True,
        "last_confirmed": None,
        "repeat_count": 0,
        "lock": False,
        "lock_start_time": 0
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "call_started" not in st.session_state:
    st.session_state.call_started = False
if "end_call" not in st.session_state:
    st.session_state.end_call = False

col1, col2, col3 = st.columns([2.5, 2.5, 1.5])

with col1:
    st.markdown("#### You")
    cam_feed = st.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    pad_col, btn_col = st.columns([8, 1])
    with btn_col:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

        if st.button("📷", key="btn_camera"):
            st.session_state.camera_on = not st.session_state.camera_on

with col2:
    st.markdown("#### Remote")
    st.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

    if "call_started" not in st.session_state:
        st.session_state.call_started = False
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False
    if "end_call" not in st.session_state:
        st.session_state.end_call = False

    button_label = "📞End Call" if st.session_state.call_started else "📞Start Call"

    if st.button(button_label, key="btn_toggle_call"):
        if not st.session_state.call_started:
            #Start call
            st.session_state.call_started = True
            st.session_state.camera_on = True
            st.session_state.end_call = False
        else:
            #End call
            st.session_state.call_started = False
            st.session_state.camera_on = False
            st.session_state.end_call = False

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("### AI Recognition")
    gesture_text = st.empty()
    percentage_text = st.empty()
    conf_bar = st.empty()
    # st.markdown("---")
    # st.markdown("### Live Translation")
    col_btns = st.columns([1, 1])
    with col_btns[0]:
        if st.button("📋 Copy"):
            text = " ".join([g for g, _ in st.session_state.history[-6:]])
            pyperclip.copy(text)
            st.session_state.copy_success = True
    with col_btns[1]:
        if st.button("🗑 Clear"):
            st.session_state.history = []

    if st.session_state.copy_success:
        st.success("Copied successfully!")
        st.session_state.copy_success = False
    
    live_output = st.empty()

    st.markdown("---")
    st.markdown("### Translation History")
    history_box = st.empty()

cap = cv2.VideoCapture(0)

def should_add_gesture(current):
    cooldown_seconds = 2.5

    if st.session_state.lock:
        if time.time() - st.session_state.lock_start_time < cooldown_seconds:
            return False
        else:
            st.session_state.lock = False

    last = st.session_state.last_confirmed

    if current != last:
        st.session_state.last_confirmed = current
        st.session_state.lock = True
        st.session_state.lock_start_time = time.time()
        return True

    st.session_state.lock = True
    st.session_state.lock_start_time = time.time()
    return True

try:
    while cap.isOpened() and not st.session_state.end_call:
        if st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠ Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = ""
            confidence = 0
            motion_label = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

                    if len(landmarks) == 63:
                        pred = smodel.predict(np.array([landmarks]))[0]
                        class_id = np.argmax(pred)
                        confidence = pred[class_id]
                        # gesture = s_encoder.inverse_transform([class_id])[0]

                        if confidence > 0.85:
                            gesture = s_encoder.inverse_transform([class_id])[0]
                            now = time.time()

                            if gesture == st.session_state.gesture_candidate:
                                if now - st.session_state.gesture_timer > 3.0:
                                    if should_add_gesture(gesture):
                                        st.session_state.stable_gesture = gesture
                                        st.session_state.history.append((gesture, min(int(confidence * 100), 100)))
                                        play_mp3_for_gesture(gesture)
                                        threading.Thread(target=speak, args=(gesture,), daemon=True).start()
                            else:
                                st.session_state.gesture_candidate = gesture
                                st.session_state.gesture_timer = now

                        st.session_state.motion_buffer.append(landmarks)
                        if st.session_state.motion_buffer:
                            if len(st.session_state.motion_buffer) > SEQUENCE_LENGTH:
                                st.session_state.motion_buffer.pop(0)


                        if len(st.session_state.motion_buffer) == SEQUENCE_LENGTH:
                            sequence = np.array([st.session_state.motion_buffer])
                            mpred = mmodel.predict(sequence)[0]
                            m_class = np.argmax(mpred)
                            motion_label = m_encoder.inverse_transform([m_class])[0]

                            if motion_label != st.session_state.last_spoken_label and (now - st.session_state.last_spoken_time) > st.session_state.cooldown:
                                threading.Thread(target=speak, args=(motion_label,), daemon=True).start()
                                st.session_state.last_spoken_label = motion_label
                                st.session_state.last_spoken_time = now
                                st.session_state.stable_gesture = motion_label
                                st.session_state.history.append((motion_label, 100))

                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cam_feed.image(frame, channels="BGR")

            
            if st.session_state.stable_gesture == gesture:
                percentage = min(int(confidence * 100), 100)
                gesture_text.markdown(f"Gesture: {gesture}")
                percentage_text.markdown(f"{percentage}%")
                conf_bar.progress(percentage)
                # conf_bar.progress(min(int(confidence * 100), 100))
                # if not st.session_state.history or gesture != st.session_state.history[-1]:
                #     st.session_state.history.append(gesture)

            elif st.session_state.stable_gesture == motion_label:
                percentage = min(int(confidence * 100), 100)
                gesture_text.markdown(f"Gesture: {motion_label}")
                percentage_text.markdown(f"{percentage}%")
                conf_bar.progress(percentage)
                # conf_bar.progress(min(int(confidence * 100), 100))
                # if not st.session_state.history or gesture != st.session_state.history[-1]:
                #     st.session_state.history.append(gesture)

            else:
                gesture_text.markdown("Gesture: -")
                percentage_text.markdown("")
                conf_bar.progress(0)

            live_output.markdown(" ".join([g for g, _ in st.session_state.history[-6:]]))
            history_html = "".join([
                f"<div style='line-height:1.6;'>"
                f"<span style='color:#999; font-weight:bold; font-size:0.85rem;'>{pct}%</span> "
                f"<span style='font-weight:bold; font-size:1.2rem; margin-left:12px;'>{gesture}</span>"
                f"</div>"
                for gesture, pct in reversed(st.session_state.history[-10:])
            ])

            history_box.markdown(
                f"""
                <div style='
                    max-height: 160px;
                    overflow-y: auto;
                    overflow-x: hidden;
                    padding-right: 8px;
                    border: 1px solid #444;
                    border-radius: 6px;
                    padding: 8px;
                    background-color: #111;
                '>
                    {history_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            cam_feed.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")
            gesture_text.markdown("Camera is Off")
            conf_bar.progress(0)
            live_output.markdown("")
            history_box.markdown("")

        time.sleep(0.03)

finally:
    if cap.isOpened():
        cap.release()
