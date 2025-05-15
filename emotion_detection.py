from deepface import DeepFace
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import pyttsx3
import ollama

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def generate_ai_response(emotion):
    prompt = f"Benim ruh halim şu an {emotion}. Lütfen kısa ve anlamlı bir Türkçe mesaj ver."
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        message = response['message']['content']
        return message
    except Exception as e:
        print("AI response error:", e)
        return "Üzgünüm, şu anda yanıt veremiyorum."

# --- Giriş ekranı ---
def show_intro():
    selection = {'camera': 0}
    def on_start():
        selection['camera'] = int(cam_combo.get())
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
    return selection['camera']

cam_idx = show_intro()

# Emotion colors
emotion_colors = {
    'happy': (0, 255, 0), 'sad': (255, 0, 0), 'angry': (0, 0, 255),
    'surprise': (0, 255, 255), 'fear': (128, 0, 128), 'neutral': (200, 200, 200)
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))

cap = cv2.VideoCapture(cam_idx)

frame_count = 0
emotion_result = None
last_detected_emotion = "neutral"

# --- GUI Buton ve Mesaj Paneli ---
def start_ai_gui():
    def on_send():
        ai_message = generate_ai_response(last_detected_emotion)
        chat_area.insert(tk.END, f"User ({last_detected_emotion}): [Sending...]\n")
        chat_area.insert(tk.END, f"AI: {ai_message}\n\n")
        chat_area.see(tk.END)
        speak(ai_message)

    gui = tk.Tk()
    gui.title("AI Control Panel with Chat")
    gui.geometry("400x300")

    ttk.Label(gui, text="Last Detected:").pack()
    detected_label = ttk.Label(gui, text=last_detected_emotion.upper(), font=('Arial', 12))
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
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            emotion_result = result
            last_detected_emotion = result[0]['dominant_emotion']
    except Exception as e:
        print("Error:", e)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    if frame_count % 3 == 0:
        threading.Thread(target=analyze_emotion, args=(frame,)).start()

    if emotion_result:
        for face in emotion_result:
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            color = emotion_colors.get(last_detected_emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, last_detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Sadece kamera görüntüsü göster, sanal yüzü kaldırdık
    cv2.imshow("Emotion Analysis & AI Panel", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
