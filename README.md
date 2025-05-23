# 🤖 Emotion & Gesture Recognition with AI-Enhanced Emojis

### by **Mehmet Hakan Terzi** & **Duygu Ayvaz**  
_Sesinizden duygunuzu anlayan, ifadenizi analiz eden akıllı bir sistem_

---

## 💡 Projenin Özeti

Bu proje; bilgisayarlı görüş, yapay zeka ve jest tanıma tekniklerini bir araya getirerek **duygularınızı ve el hareketlerinizi anlayan**, anladığını emojiyle gösteren ve hatta ruh halinize özel anlamlı cümleler üreten bir sistem sunar.

📷 Kamera açılır, 🎭 yüzünüz okunur, ✋ eliniz analiz edilir ve 🧠 yapay zeka size özel tepki verir.  
**Bu bir yazılım değil; bu, sizinle iletişim kuran bir deneyim.**

---

## 🧰 Kullanılan Teknolojiler

- `OpenCV` – Gerçek zamanlı video işleme
- `MediaPipe` – El ve yüz algılama
- `DeepFace` – Yüz ifadelerinden duygu analizi
- `pyttsx3` – Sesli yanıt üretimi
- `Ollama` + `llama3.2` – AI ile duygu temelli cümle üretimi
- `Tkinter` – Kamera ve AI paneli için GUI

---

## 📂 Proje Yapısı

```
.
├── emojis/
│   ├── happy.png, sad.png, angry.png, ...
├── emotion_detection.py
└── README.md
```

> `emojis/` klasörü, tanınan jest ve duygulara karşılık gelen görselleri içerir.

---

## 🔧 Kurulum

### 🐍 Gereksinimler:

```bash
pip install opencv-python mediapipe numpy pyttsx3 deepface ollama
```

> ⚠️ `ollama` sisteminizde kurulu olmalı ve `llama3.2` modeli hazır durumda bulunmalıdır.

---

## 🧪 Nasıl Çalışır?

1. Uygulama başladığında bir arayüz açılır ve kameranızı seçersiniz.
2. Kamera görüntüsü üzerinde yüz ifadeleri ve el hareketleri analiz edilir.
3. Tanınan duygu ve jest, uygun emojiyle görselleştirilir.
4. "AI'ya Gönder" butonuna basıldığında, son tespit edilen duygu AI’ya iletilir.
5. AI, duyguya uygun anlamlı bir Türkçe cümle üretir ve sesli olarak okunur.

---

## 😄 Desteklenen Duygular

- Happy 😊  
- Sad 😢  
- Angry 😠  
- Surprise 😮  
- Fear 😱  
- Neutral 😐

---

## ✋ Tanınan El Jestleri

| Jest        | Anlamı              |
|-------------|---------------------|
| LIKE        | 👍 Beğeni            |
| OK          | 👌 Tamam             |
| PUNCH       | ✊ Yumruk            |
| HELLO       | 🖐️ Selam             |
| TURKIYE     | 🤟🇹🇷 Yerli jest        |
| LOVE_JP     | ❤️ Kalp işareti      |

---

## 🧠 AI ile Konuş

Uygulama, AI motoruna son tespit edilen ruh halini şöyle iletir:

> `"Benim ruh halim şu an happy. Lütfen kısa ve anlamlı bir Türkçe mesaj ver."`

Ve gelen yanıt kullanıcıya hem yazılı hem sesli olarak sunulur.  
**Kişiselleştirilmiş bir yapay zeka deneyimi.**

---

## ⌨️ Kontroller

- `q` → Uygulamadan çıkış
- `"Send to AI"` → Yapay zekadan duyguya özel mesaj al

---

## 📜 Lisans

Bu yazılım; **duyguyu, insan-makine etkileşimini ve teknolojiyi seven herkes için özgürdür**.  
Ticari veya kişisel kullanımda dilediğiniz gibi faydalanabilirsiniz.   

---

## 📫 İletişim

 Mehmet Hakan Terzi  
📧 trz.hakanterzi@gmail.com

 Duygu Ayvaz  
📧 duygu.ayvaz.tr@gmail.com

---

