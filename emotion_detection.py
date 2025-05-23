import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import pyttsx3
import ollama
from deepface import DeepFace
from collections import deque

def flip_x(x):
    return 1 - x

def overlay_image(background, overlay, x, y, target_size=None):
    if overlay is None:
        print("Uyarı: PNG yüklenemedi.")
        return background
    if target_size:
        ov = cv2.resize(overlay, target_size)
    else:
        h_bg, _, _ = background.shape
        h_ov, w_ov, _ = overlay.shape
        scale = h_bg // 4
        ratio = w_ov / h_ov
        ov = cv2.resize(overlay, (int(scale * ratio), scale))
    h, w, _ = ov.shape
    rows, cols, _ = background.shape
    if y + h > rows or x + w > cols:
        return background
    roi = background[y:y+h, x:x+w]
    gray = cv2.cvtColor(ov, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(ov, ov, mask=mask)
    background[y:y+h, x:x+w] = cv2.add(bg, fg)
    return background

emoji_images = {
    "happy":    cv2.imread("emojis/happy.png"),
    "sad":      cv2.imread("emojis/sad.png"),
    "angry":    cv2.imread("emojis/angry.png"),
    "surprise": cv2.imread("emojis/surprise.png"),
    "fear":     cv2.imread("emojis/fear.png"),
    "neutral":  cv2.imread("emojis/neutral.png"),
    "LIKE":     cv2.imread("emojis/like.png"),
    "OK":       cv2.imread("emojis/ok.png"),
    "PUNCH":    cv2.imread("emojis/punch.png"),
    "HELLO":    cv2.imread("emojis/hello.png"),
    "TURKIYE":  cv2.imread("emojis/turkiye.png"),
    "LOVE_JP":  cv2.imread("emojis/heart.png"),
}

engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

def generate_ai_response(emotion):
    prompt = f"My mood right now is {emotion}. Please give me a short and meaningful message for my mood."
    try:
        res = ollama.chat(model='llama3.2', messages=[{'role':'user','content':prompt}])
        return res['message']['content']
    except Exception as e:
        print("AI error:", e)
        return "Üzgünüm, şu anda yanıt veremiyorum."

def show_intro():
    sel = {'camera': 0}
    def on_start():
        sel['camera'] = int(cam_combo.get())
        root.destroy()
    root = tk.Tk()
    root.title("Emotion-Detection Settings")
    root.geometry("300x120")
    ttk.Label(root, text="Select Camera:").pack(pady=(10,0))
    cam_combo = ttk.Combobox(root, values=[0,1,2,3,4], state="readonly")
    cam_combo.current(0)
    cam_combo.pack()
    ttk.Button(root, text="Start", command=on_start).pack(pady=15)
    root.mainloop()
    return sel['camera']

cam_idx = show_intro()

emotion_colors = {
    'happy': (0,255,0), 'sad': (255,0,0), 'angry': (0,0,255),
    'surprise': (0,255,255), 'fear': (128,0,128), 'neutral': (200,200,200)
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.6, min_tracking_confidence=0.6)

DEBUG_DRAW_HANDS = False

cap = cv2.VideoCapture(cam_idx)
frame_count = 0
emotion_result = None
last_detected_emotion = "neutral"

EMO_HISTORY = deque(maxlen=9)
EMO_MIN_MAJORITY = 5

gesture_history = []
HISTORY_LEN = 12
STABLE_THRESHOLD = 4

def start_ai_gui():
    def on_send():
        msg = generate_ai_response(last_detected_emotion)
        chat_area.insert(tk.END, f"User ({last_detected_emotion}): [Sending...]\n")
        chat_area.insert(tk.END, f"AI: {msg}\n\n")
        chat_area.see(tk.END)
        speak(msg)
    gui = tk.Tk()
    gui.title("AI Control Panel with Chat")
    gui.geometry("400x300")
    ttk.Label(gui, text="Last Detected:").pack()
    detected_label = ttk.Label(gui, text=last_detected_emotion.upper(), font=('Arial',12))
    detected_label.pack(pady=(0,5))
    chat_area = scrolledtext.ScrolledText(gui, width=50, height=10)
    chat_area.pack(pady=(5,10))
    ttk.Button(gui, text="Send to AI", command=on_send).pack()
    def update_label():
        detected_label.config(text=last_detected_emotion.upper())
        gui.after(500, update_label)
    update_label()
    gui.mainloop()

threading.Thread(target=start_ai_gui, daemon=True).start()

def analyze_emotion(frame):
    global emotion_result, last_detected_emotion
    try:
        res = DeepFace.analyze(
            frame, actions=['emotion'],
            enforce_detection=False) # geri orijinale döndü
        if isinstance(res, list):
            emotion_result = res
            dom = res[0]['dominant_emotion']
            EMO_HISTORY.append(dom)
            if EMO_HISTORY.count(dom) >= EMO_MIN_MAJORITY:
                last_detected_emotion = dom
    except Exception as e:
        print("DeepFace error:", e)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    fh, fw, _ = frame.shape

    # Emotion analizini her 3 frame’de bir çalıştır
    if frame_count % 3 == 0:
        threading.Thread(target=analyze_emotion, args=(frame,)).start()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

    extra = np.zeros((fh, 600, 3), dtype=np.uint8)
    frame_with_space = np.hstack((frame, extra))

    cv2.putText(frame_with_space, "DETECTED EMOTIONS & GESTURES",
                (fw+10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # --- YÜZ ve EMOJI ---
    if emotion_result:
        for face in emotion_result:
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            col = emotion_colors.get(last_detected_emotion, (255,255,255))
            cv2.rectangle(frame_with_space, (x, y), (x+w, y+h), col, 2)
            cv2.putText(frame_with_space, last_detected_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            if last_detected_emotion in emoji_images:
                frame_with_space = overlay_image(
                    frame_with_space,
                    emoji_images[last_detected_emotion],
                    fw + 50, 100
                )

    # --- EL GESTURE ve Smoothing ---
    stable_gesture = None
    if hand_results.multi_hand_landmarks:
        # Sadece ilk el
        hand_landmarks = hand_results.multi_hand_landmarks[0]

        if DEBUG_DRAW_HANDS:
            mp_drawing.draw_landmarks(frame_with_space, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        lm = hand_landmarks.landmark
        wrist_y = lm[mp_hands.HandLandmark.WRIST].y

        def is_extended(tip, pip, th=0.035):
            return lm[tip].y < lm[pip].y - th

        idx_ext = is_extended(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
        mid_ext = is_extended(mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
        ring_ext = is_extended(mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
        pinky_ext = is_extended(mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
        # Baş parmak toleransı biraz daha yüksek
        thumb_ext = lm[mp_hands.HandLandmark.THUMB_TIP].y < wrist_y - 0.025

        dist_heart = np.linalg.norm([
            (1 - lm[mp_hands.HandLandmark.THUMB_TIP].x) -
            (1 - lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x),
            lm[mp_hands.HandLandmark.THUMB_TIP].y -
            lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])

        gesture = None
        # TÜRKİYE: Orta ve yüzük kesin kıvrık, işaret ve pinky açık, baş parmak biraz toleranslı (çok açık değilse)
        if idx_ext and pinky_ext and thumb_ext and (not mid_ext) and (not ring_ext):
            gesture = "TURKIYE"
        # LOVE (Japon kalp)
        elif dist_heart < 0.06 and idx_ext and thumb_ext:
            gesture = "LOVE_JP"
        # HELLO
        elif idx_ext and mid_ext and ring_ext and pinky_ext:
            gesture = "HELLO"
        # PUNCH
        elif not (idx_ext or mid_ext or ring_ext or pinky_ext or thumb_ext):
            gesture = "PUNCH"
        # LIKE
        elif thumb_ext and not (idx_ext or mid_ext or ring_ext or pinky_ext):
            gesture = "LIKE"
        # OK
        elif np.linalg.norm([
                (1 - lm[mp_hands.HandLandmark.THUMB_TIP].x) -
                (1 - lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x),
                lm[mp_hands.HandLandmark.THUMB_TIP].y -
                lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y]) < 0.04 \
             and mid_ext and ring_ext and pinky_ext:
            gesture = "OK"

        gesture_history.append(gesture)
        if len(gesture_history) > HISTORY_LEN:
            gesture_history.pop(0)

        if gesture and gesture_history.count(gesture) >= STABLE_THRESHOLD:
            stable_gesture = gesture
        if stable_gesture and stable_gesture != gesture_history[-1]:
            gesture_history.clear()

    if stable_gesture and stable_gesture in emoji_images:
        frame_with_space = overlay_image(
            frame_with_space,
            emoji_images[stable_gesture],
            fw + 50, 300)

    fps = int(1/(time.time() - prev_time))
    cv2.putText(frame_with_space, f"FPS: {fps}",
                (10, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    prev_time = time.time()

    cv2.imshow("Emotion & Gesture Recognition with PNG Emojis", frame_with_space)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
