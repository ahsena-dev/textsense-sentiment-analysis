# ============================================================
# BIM 430 — Derin Öğrenme Proje
# Türkçe Duygu Analizi (Turkish Sentiment Analysis)
# 3 Model: Custom MLP | Bidirectional LSTM | TextCNN
# ============================================================

# ── 0. KURULUM ──────────────────────────────────────────────
# pip install tensorflow datasets scikit-learn matplotlib seaborn

import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ── 1. GPU / SEED AYARLARI ──────────────────────────────────
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✅ GPU aktif: {gpus[0].name}")
    # Mixed precision → GPU'da ~2x hız
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    BATCH_SIZE = 128   # GPU: büyük batch
else:
    print("⚠️  CPU modu — eğitim yavaş olabilir")
    BATCH_SIZE = 64    # CPU: küçük batch

from tensorflow import keras
from tensorflow.keras import layers

# ── 2. HİPERPARAMETRELER ────────────────────────────────────
MAX_WORDS  = 30_000   # Colab'dan büyük vocab
MAX_LEN    = 150      # Colab'dan daha uzun sekans
EMBED_DIM  = 128      # Colab'dan 2x büyük embedding
EPOCHS     = 15

print(f"\nConfig → vocab:{MAX_WORDS} | maxlen:{MAX_LEN} | embed:{EMBED_DIM} | batch:{BATCH_SIZE}")

# ── 3. VERİ YÜKLEME ─────────────────────────────────────────
print("\n📥 Dataset yükleniyor...")
from datasets import load_dataset

dataset = load_dataset('winvoker/turkish-sentiment-analysis-dataset')
df_raw  = dataset['train'].to_pandas()

print(f"Ham veri: {df_raw.shape}")
print(df_raw['label'].value_counts())

# Sadece Positive ve Negative → ikili sınıflandırma
df = df_raw[df_raw['label'].isin(['Positive', 'Negative'])].copy()
df_pos = df[df['label'] == 'Positive'].sample(50_000, random_state=SEED)
df_neg = df[df['label'] == 'Negative'].sample(50_000, random_state=SEED)
df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
df['label'] = df['label'].map({'Positive': 1, 'Negative': 0})

print(f"\n✅ Kullanılan veri: {df.shape}  |  Dengeli: {df['label'].value_counts().to_dict()}")

# ── 4. ÖN İŞLEME ────────────────────────────────────────────
import re

def temizle(metin):
    metin = str(metin).lower()
    metin = re.sub(r'http\S+|www\S+', '', metin)   # URL kaldır
    metin = re.sub(r'[^\w\s]', ' ', metin)          # Noktalama
    metin = re.sub(r'\d+', '', metin)               # Sayılar
    metin = re.sub(r'\s+', ' ', metin).strip()
    return metin

df['text_clean'] = df['text'].apply(temizle)
print(f"\nÖrnek temizleme:")
print(f"  ORİJİNAL : {df['text'].iloc[0][:80]}")
print(f"  TEMİZ    : {df['text_clean'].iloc[0][:80]}")

# ── 5. TOKENİZASYON & PADDİNG ───────────────────────────────
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text_clean'])

sequences = tokenizer.texts_to_sequences(df['text_clean'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = df['label'].values

print(f"\nX: {X.shape}  |  y: {y.shape}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print(f"Eğitim: {X_train.shape[0]} | Validasyon: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ── 6. MODELLER ──────────────────────────────────────────────

# ── Model 1: STUDENT-DESIGNED Custom MLP ────────────────────
# Mimari tasarımı:
#   Embedding → Conv1D (özellik çıkarımı) → GlobalAvg+GlobalMax birleştirme
#   (Ortalama + Maksimum bilgiyi birleştirmek MLP'nin kelime önemine duyarlı olmasını sağlar)
#   → Dense bloklar → Sigmoid
def build_custom_mlp():
    inp = keras.Input(shape=(MAX_LEN,))

    x = layers.Embedding(MAX_WORDS, EMBED_DIM)(inp)

    # İki farklı pooling → concat (öğrenci tasarımı karar)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])   # 256 boyut

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inp, out, name='Custom_MLP_Student')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── Model 2: Bidirectional LSTM ─────────────────────────────
# Literatür modeli — iki yönlü LSTM hem ileri hem geri bağlamı yakalar
def build_bilstm():
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(MAX_WORDS, EMBED_DIM)(inp)
    x   = layers.SpatialDropout1D(0.2)(x)

    x   = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x   = layers.Bidirectional(layers.LSTM(64,  dropout=0.2))(x)

    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inp, out, name='BiLSTM')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── Model 3: TextCNN (multi-kernel) ─────────────────────────
# Literatür modeli — Kim (2014) mimarisi
# 3, 4, 5 kelimelik n-gramları paralel filtrelerle yakalar
def build_textcnn():
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(MAX_WORDS, EMBED_DIM)(inp)

    # Farklı boyutlarda filtreler
    convs = []
    for ks in [3, 4, 5]:
        c = layers.Conv1D(128, kernel_size=ks, activation='relu', padding='same')(x)
        c = layers.GlobalMaxPooling1D()(c)
        convs.append(c)

    x   = layers.Concatenate()(convs)   # 3×128 = 384 özellik
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inp, out, name='TextCNN')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── 7. CALLBACK'LER ──────────────────────────────────────────
def get_callbacks(name):
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=4,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'best_{name}.keras', monitor='val_accuracy',
            save_best_only=True, verbose=0
        ),
    ]

# ── 8. EĞİTİM ───────────────────────────────────────────────
fit_kwargs = dict(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
)

print("\n" + "="*55)
print("=== Model 1: Custom MLP (Öğrenci Tasarımı) ===")
mlp_model   = build_custom_mlp()
mlp_model.summary()
mlp_history = mlp_model.fit(X_train, y_train,
                             callbacks=get_callbacks('mlp'), **fit_kwargs)

print("\n" + "="*55)
print("=== Model 2: Bidirectional LSTM ===")
bilstm_model   = build_bilstm()
bilstm_model.summary()
bilstm_history = bilstm_model.fit(X_train, y_train,
                                   callbacks=get_callbacks('bilstm'), **fit_kwargs)

print("\n" + "="*55)
print("=== Model 3: TextCNN (Kim 2014) ===")
cnn_model   = build_textcnn()
cnn_model.summary()
cnn_history = cnn_model.fit(X_train, y_train,
                             callbacks=get_callbacks('cnn'), **fit_kwargs)

# ── 9. DEĞERLENDİRME ────────────────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

def evaluate_model(model, X_test, y_test, name):
    y_pred = (model.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(classification_report(y_test, y_pred,
                                  target_names=['Negatif', 'Pozitif']))
    return {
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_test, y_pred),  4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred),    4),
        'F1-Score':  round(f1_score(y_test, y_pred),        4),
        'CM':        confusion_matrix(y_test, y_pred),
        'y_pred':    y_pred,
    }

print("\n\n" + "="*55)
print("TEST SETİ SONUÇLARI")
results = [
    evaluate_model(mlp_model,    X_test, y_test, 'Custom MLP (Öğrenci)'),
    evaluate_model(bilstm_model, X_test, y_test, 'Bidirectional LSTM'),
    evaluate_model(cnn_model,    X_test, y_test, 'TextCNN'),
]

results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['CM', 'y_pred']}
                             for r in results])
print("\n\n📊 ÖZET TABLO:")
print(results_df.to_string(index=False))

# ── 10. GRAFİKLER ───────────────────────────────────────────
os.makedirs('grafik', exist_ok=True)

COLORS = ['#e74c3c', '#3498db', '#27ae60']
MODEL_NAMES  = ['Custom MLP', 'BiLSTM', 'TextCNN']
HISTORIES    = [mlp_history, bilstm_history, cnn_history]

# 10a. Training Loss + Accuracy
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for hist, name, color in zip(HISTORIES, MODEL_NAMES, COLORS):
    axes[0].plot(hist.history['loss'],     color=color, lw=2, label=f'{name} Train')
    axes[0].plot(hist.history['val_loss'], color=color, lw=2, ls='--', label=f'{name} Val')
    axes[1].plot(hist.history['accuracy'],     color=color, lw=2, label=f'{name} Train')
    axes[1].plot(hist.history['val_accuracy'], color=color, lw=2, ls='--', label=f'{name} Val')

axes[0].set_title('Eğitim Kaybı (Training Loss)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].set_title('Doğruluk Grafiği (Accuracy)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('grafik/egitim_grafikleri.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅  grafik/egitim_grafikleri.png kaydedildi")

# 10b. Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cmaps = ['Reds', 'Blues', 'Greens']
for ax, result, name, cmap in zip(axes, results, MODEL_NAMES, cmaps):
    sns.heatmap(result['CM'], annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Negatif', 'Pozitif'],
                yticklabels=['Negatif', 'Pozitif'],
                annot_kws={'size': 13})
    ax.set_title(f'{name}\nAcc: {result["Accuracy"]} | F1: {result["F1-Score"]}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gerçek')

plt.suptitle('Confusion Matrix Karşılaştırması', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('grafik/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅  grafik/confusion_matrix.png kaydedildi")

# 10c. Metrik bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x     = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, (result, color) in enumerate(zip(results, COLORS)):
    vals = [result[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width,
                  label=result['Model'], color=color, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_title('Model Performans Karşılaştırması', fontsize=15, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0.80, 1.01)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('grafik/metrik_karsilastirma.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅  grafik/metrik_karsilastirma.png kaydedildi")

# ── 11. DEMO: CANLI TAHMİN ───────────────────────────────────
def tahmin_et(yorum):
    temiz  = temizle(yorum)
    seq    = tokenizer.texts_to_sequences([temiz])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    mlp_p    = mlp_model.predict(padded,    verbose=0)[0][0]
    bilstm_p = bilstm_model.predict(padded, verbose=0)[0][0]
    cnn_p    = cnn_model.predict(padded,    verbose=0)[0][0]

    etiket = lambda p: '😊 POZİTİF' if p >= 0.5 else '😡 NEGATİF'
    print(f'\nYorum   : "{yorum}"')
    print(f'MLP     : {etiket(mlp_p)}    ({mlp_p:.3f})')
    print(f'BiLSTM  : {etiket(bilstm_p)} ({bilstm_p:.3f})')
    print(f'TextCNN : {etiket(cnn_p)}    ({cnn_p:.3f})')
    print('─' * 50)

print("\n\n=== DEMO TAHMİNLER ===")
tahmin_et('Ürün çok kaliteliydi, çok memnun kaldım!')
tahmin_et('Berbat bir ürün, param çöp oldu')
tahmin_et('İdare eder, fiyatına göre normal')
tahmin_et('Harika paketleme, hızlı kargo süper!')
tahmin_et('Hiç beğenmedim, iade ettim')

# ── 12. MODELLERİ KAYDET ────────────────────────────────────
import pickle

mlp_model.save('model_mlp.keras')
bilstm_model.save('model_bilstm.keras')
cnn_model.save('model_cnn.keras')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\n✅  Tüm modeller ve tokenizer kaydedildi.")
print("    model_mlp.keras | model_bilstm.keras | model_cnn.keras | tokenizer.pkl")
