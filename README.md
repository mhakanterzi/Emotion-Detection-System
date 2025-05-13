# Emotion Detection System

Basit bir DeepFace + MediaPipe tabanlı Python uygulaması.

## Özellikler
- 🔍 **Gerçek zamanlı yüz tespiti** (MediaPipe Face Mesh)  
- 😊 **Duygu analizi** (DeepFace)  
- 🎶 **Ruh haline göre şarkı önerisi**  
- 💬 **Ruh haline göre komik/rahatlatıcı söz gösterimi**  
- ⚙️ **Kamera seçimli, esnek bir GUI** (Tkinter)  
- 📊 **FPS ve duygu geçmişi için smoothing** (opsiyonel) 

## Gereksinimler
- **Python** ≥ 3.7  
- **pip**  
- **Git**  

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
    ```
    1. Açılan Tkinter penceresinde  
   - Kullanmak istediğiniz **kamera numarasını** seçin  
   - “Şarkı öner” ve/veya “Söz göster” seçeneklerini işaretleyin  
   - **Başlat** butonuna tıklayın  

    2. Yeni pencerede  
   - Gerçek zamanlı duygu analizi ve yüz mesh görselleştirmesini izleyin  
   - Önerilen şarkı ve sözler ekranın üstünde belirecektir  
   - Çıkmak için \`q\` tuşuna basın  
