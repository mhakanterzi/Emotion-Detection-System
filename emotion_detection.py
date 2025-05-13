from deepface import DeepFace
import cv2
import mediapipe as mp
import numpy as np
import time
import threading

### GEREKLİ KÜTÜPHANELERİ EKLİYORUZ
import tkinter as tk
from tkinter import ttk

# --- YENİ: Giriş ekranı fonksiyonu ---
def show_intro():
    # Kullanıcı tercihlerinin saklanacağı değişkenler
    selection = {'camera': 0, 'songs': False, 'quotes': False}
    
    def on_start():
        # Combobox'tan ve checkbutton'lardan seçimi al
        selection['camera'] = int(cam_combo.get())
        selection['songs']  = var_songs.get()
        selection['quotes'] = var_quotes.get()
        root.destroy()

    root = tk.Tk()
    root.title("Emotion-Detection Ayarları")
    root.geometry("300x200")
    root.resizable(False, False)

    ttk.Label(root, text="Kamera Seçin:").pack(pady=(10,0))
    cam_combo = ttk.Combobox(root, values=[0,1,2,3,4], state="readonly")
    cam_combo.current(0)
    cam_combo.pack()

    var_songs  = tk.BooleanVar(value=False)
    var_quotes = tk.BooleanVar(value=False)
    ttk.Checkbutton(root, text="Ruh haline göre rastgele şarkı öner", variable=var_songs).pack(pady=5)
    ttk.Checkbutton(root, text="Ruh haline göre komik/rahatlatıcı söz göster", variable=var_quotes).pack()

    ttk.Button(root, text="Başlat", command=on_start).pack(pady=15)
    root.mainloop()

    return selection['camera'], selection['songs'], selection['quotes']
# --- /YENİ ---

# --- YENİ: Şarkı ve söz listeleri ---
emotion_songs = {
    'happy':   ["Here Comes the Sun","Happy","I Gotta Feeling"],
    'sad':     ["Someone Like You","Fix You","Yesterday"],
    'angry':   ["Breaking the Law","Killing in the Name","Smells Like Teen Spirit"],
    'surprise':["Surprise Yourself","What a Wonderful World"],
    'fear':    ["Thriller","Disturbia"],
    'disgust': ["Dirty","Bad","U Can't Touch This"],
    'neutral':["Let It Be","Comfortably Numb"]
}

emotion_quotes = {
    'happy':   "Gülümse, dünya seninle güzel 😊",
    'sad':     "Her fırtınadan sonra gökkuşağı çıkar 🌈",
    'angry':   "Derin bir nefes al, sakinleş 🌬️",
    'surprise':"Hayat sürprizlerle dolu 🎉",
    'fear':    "Korku, cesaretin başlangıcıdır 💪",
    'disgust': "Negatiflikten sıyrıl, pozitif kal ✨",
    'neutral':"Sakinlik huzurun anahtarıdır 🗝️"
}
# --- /YENİ ---

# --- Şimdi kullanıcıdan ayarları alıyoruz ---
cam_idx, use_songs, use_quotes = show_intro()

# MediaPipe yüz mesh modülü
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing   = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))

# Duyguya göre renkler
emotion_colors = {
    'happy':    (0, 255, 0),
    'sad':      (255, 0, 0),
    'angry':    (0, 0, 255),
    'surprise': (0, 255, 255),
    'fear':     (128, 0, 128),
    'disgust':  (0, 128, 0),
    'neutral':  (200, 200, 200)
}

# Kamera başlat (seçilen idx)
cap = cv2.VideoCapture(cam_idx)
prev_time = time.time()

frame_count     = 0
emotion_result  = None
last_emotion    = None
suggested_song  = ""
suggested_quote = ""

def analyze_emotion(frame):
    global emotion_result
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            emotion_result = result
    except Exception as e:
        print("Hata:", e)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Aynalı görüntü
    frame = cv2.flip(frame, 1)

    # FPS hesapla
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Her 5 karede bir analiz
    if frame_count % 5 == 0:
        threading.Thread(target=analyze_emotion, args=(frame,)).start()

    # Duygu sonucu varsa
    if emotion_result:
        for face in emotion_result:
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            emotion = face['dominant_emotion']
            color   = emotion_colors.get(emotion, (255, 255, 255))

            # Yüz bölgesi ve duygu etiketi
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # --- YENİ: Şarkı önerisi ---
            if use_songs and emotion != last_emotion:
                suggested_song = np.random.choice(emotion_songs.get(emotion, []))
                last_emotion = emotion

            # --- YENİ: Söz önerisi ---
            if use_quotes:
                suggested_quote = emotion_quotes.get(emotion, "")

    # MediaPipe ile yüz hatları
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_res  = face_mesh.process(rgb_frame)
    face_canvas = np.zeros_like(frame)
    if mesh_res.multi_face_landmarks:
        for lm in mesh_res.multi_face_landmarks:
            mp_drawing.draw_landmarks(face_canvas, lm, mp_face_mesh.FACEMESH_TESSELATION,
                                      drawing_spec, drawing_spec)

    # Önerileri ekrana yaz
    if use_songs and suggested_song:
        cv2.putText(frame, f"Şarkı: {suggested_song}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if use_quotes and suggested_quote:
        cv2.putText(frame, suggested_quote, (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    combined = np.hstack((frame, face_canvas))
    cv2.imshow("Duygu Analizi ve Yüz Hatlari", combined)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
