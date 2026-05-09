# TextSense — Türkçe Duygu Analizi
### BIM 430 Derin Öğrenme Proje Ödevi

---

## 📌 Proje Hakkında

Bu proje, Türkçe müşteri yorumlarını **Pozitif** ve **Negatif** olarak sınıflandıran bir duygu analizi sistemidir. Üç farklı derin öğrenme modeli eğitilmiş, sonuçlar karşılaştırılmış ve geliştirilmiş versiyonları üretilmiştir. Sistem ayrıca **TextSense** adlı canlı bir web arayüzüne entegre edilmiştir.

---

## 📂 Veri Seti

| Özellik | Detay |
|---|---|
| Kaynak | [winvoker/turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) |
| Toplam boyut | 440.679 örnek |
| Kullanılan | 100.000 örnek (50.000 Pozitif + 50.000 Negatif) |
| Dil | Türkçe |
| Sınıf | Binary (Pozitif / Negatif) |

**Veri Ön İşleme:**
- Küçük harfe çevirme
- URL, noktalama ve sayı temizleme
- Tokenizasyon (MAX_WORDS: 30.000, MAX_LEN: 150)
- Train / Val / Test: %70 / %15 / %15

---

## 🧠 Modeller

### Baseline Modeller

#### 1. Custom MLP (Öğrenci Tasarımı)
- Embedding → GlobalAvgPool + GlobalMaxPool → Concat → Dense(256) → Dense(128) → Dense(64) → Sigmoid
- GlobalAverage ve GlobalMax pooling birleştirilerek hem genel ton hem güçlü duygu kelimeleri yakalanır
- Aktivasyon: ReLU + Dropout

#### 2. Bidirectional LSTM (Literatür)
- Embedding → SpatialDropout → BiLSTM(128) → BiLSTM(64) → Dense → Sigmoid
- İki yönlü LSTM hem ileri hem geri bağlamı öğrenir

#### 3. TextCNN (Literatür — Kim 2014)
- Embedding → Conv1D (kernel: 3,4,5) → GlobalMaxPool → Concat → Dense → Sigmoid
- Farklı boyutlarda n-gram örüntülerini paralel olarak yakalar

---

### Geliştirilmiş Modeller

#### 1. Custom MLP v2 (Öğrenci Tasarımı — İyileştirme)
- BatchNormalization kaldırıldı (eğitim kararsızlığı yapıyordu)
- L2 regularization eklendi
- ReLU → LeakyReLU (ölü nöron sorununu önler)
- Dropout artırıldı: 0.5 → 0.4 → 0.3

#### 2. Attention-BiLSTM (Literatür — İyileştirme)
- BiLSTM üzerine Attention katmanı eklendi
- Model artık her kelimeye önem skoru vererek ağırlıklı toplama yapar
- Önemli duygu kelimelerine odaklanır

#### 3. TextCNN v2 (Literatür — İyileştirme)
- Kernel sayısı 3 → 4 (2,3,4,5)
- Her kernel için GlobalMaxPool + GlobalAvgPool birleştirildi
- Conv katmanlarına L2 regularization eklendi

---

## 📊 Sonuçlar

### Baseline

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Custom MLP | 0.8741 | 0.8453 | 0.9157 | 0.8791 |
| BiLSTM | 0.8906 | 0.8859 | 0.8967 | 0.8913 |
| TextCNN | 0.8896 | 0.8783 | 0.9045 | 0.8912 |

### Geliştirilmiş

| Model | Accuracy | F1-Score | Değişim |
|---|---|---|---|
| Custom MLP v2 | 0.8948 | 0.8946 | ↑ +0.0155 ✅ |
| Attention-BiLSTM | 0.8893 | 0.8896 | ≈ aynı |
| TextCNN v2 | 0.8971 | 0.8975 | ↑ +0.0063 ✅ |

---

## 🗂️ Dosya Yapısı

```
BIM430_Proje/
├── bim430_proje.py              # Baseline modeller (eğitim + değerlendirme)
├── bim430_gelistirilmis.py      # Geliştirilmiş modeller
├── app.py                       # FastAPI backend
├── index.html                   # TextSense web arayüzü
├── tokenizer.pkl                # Eğitilmiş tokenizer
├── model_mlp_v2.keras           # Custom MLP v2
├── model_attn_bilstm.keras      # Attention-BiLSTM
├── model_cnn_v2.keras           # TextCNN v2
└── grafik/
    ├── egitim_grafikleri.png
    ├── confusion_matrix.png
    ├── metrik_karsilastirma.png
    ├── gelistirilmis_egitim.png
    ├── gelistirilmis_confusion.png
    └── baseline_vs_gelistirilmis.png
```

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install tensorflow-macos tensorflow-metal  # Apple Silicon için
pip install numpy pandas scikit-learn matplotlib seaborn datasets
pip install fastapi uvicorn
```

### Modelleri Eğit
```bash
python bim430_proje.py            # Baseline modeller
python bim430_gelistirilmis.py    # Geliştirilmiş modeller
```

### Web Arayüzünü Başlat
```bash
uvicorn app:app --reload
# Tarayıcıda aç: http://localhost:8000
```

---

## 🖥️ TextSense Arayüzü

- Tek yorum analizi (canlı tahmin)
- 3 modelin güven skorlarını bar chart ile gösterir
- Yıldız puanlama sistemi (1–5)
- CSV toplu yükleme (max 200 satır)
- Pozitif/Negatif dağılım grafiği (pie chart)
- Analiz geçmişi tablosu

---

## ⚙️ Teknik Detaylar

| Parametre | Değer |
|---|---|
| MAX_WORDS | 30.000 |
| MAX_LEN | 150 |
| EMBED_DIM | 128 |
| Optimizer | Adam (lr=3e-4) |
| Loss | Binary Crossentropy |
| Batch Size | 128 (GPU) / 64 (CPU) |
| Max Epoch | 15 |
| Early Stopping | patience=4 |
| ReduceLROnPlateau | factor=0.5, patience=2 |

---

## 👥 Grup Üyeleri

| İsim | Numara |
|---|---|
| Ahsen Sena Karaca | 030721060 |
| Saime İşlek | 030721036 |

---

*BIM 430 — Derin Öğrenme | 2025–2026 Bahar Dönemi*

*Ahsen Sena Karaca · Saime İşlek*
