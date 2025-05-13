# Emotion Detection System

Basit bir DeepFace + MediaPipe tabanlı Python uygulaması.

## Özellikler
- Kamera seçimi  
- Yüz tespiti ve gerçek zamanlı duygu analizi  
- Duyguya göre rastgele şarkı önerisi  
- Duyguya göre komik/rahatlatıcı söz gösterimi  

## Gereksinimler
- Python 3.7 veya üzeri  
- pip  
- Git  

## Kurulum

1. Repoyu klonlayın  
   ```bash
   git clone https://github.com/mhakanterzi/Emotion-Detection-System.git
   cd emotion-detection-system
2.  Sanal ortam oluşturun ve aktif edin   
    #### Windows   
    ```bash
        python -m venv deepface_env
        .\deepface_env\Scripts\activate
    ```
    #### Linux   
    ```bash
        python3 -m venv deepface_env
        source deepface_env/bin/activate
    ```
3.  Gerekli paketlerin yüklenmesi
    ```bash
    pip install -r requirements.txt
4. Uygulamanın çalıştırılması
    ```bash
    python emotion_detection.py