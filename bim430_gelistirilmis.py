# ============================================================
# BIM 430 — 3 MODELİN GELİŞTİRİLMİŞ VERSİYONLARI
#
# Baseline sonuçlar:
#   Custom MLP  → Acc: 0.8741  F1: 0.8791
#   BiLSTM      → Acc: 0.8906  F1: 0.8913
#   TextCNN     → Acc: 0.8896  F1: 0.8912
#
# Hedef: Her modeli geliştir, karşılaştır
# ============================================================

import os, warnings, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # mixed_float16 Apple Silicon'da yavaşlatıyor — kapalı
    BATCH_SIZE = 128
    print(f"✅ GPU: {gpus[0].name}")
else:
    BATCH_SIZE = 64
    print("⚠️  CPU modu")

MAX_WORDS = 30_000
MAX_LEN   = 150
EMBED_DIM = 128
EPOCHS    = 15

# ── VERİ ─────────────────────────────────────────────────────
print("\n📥 Veri yükleniyor...")
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

dataset = load_dataset('winvoker/turkish-sentiment-analysis-dataset')
df_raw  = dataset['train'].to_pandas()

df = df_raw[df_raw['label'].isin(['Positive', 'Negative'])].copy()
df = pd.concat([
    df[df['label']=='Positive'].sample(50_000, random_state=SEED),
    df[df['label']=='Negative'].sample(50_000, random_state=SEED)
]).reset_index(drop=True)
df['label'] = df['label'].map({'Positive': 1, 'Negative': 0})

def temizle(metin):
    metin = str(metin).lower()
    metin = re.sub(r'http\S+|www\S+', '', metin)
    metin = re.sub(r'[^\w\s]', ' ', metin)
    metin = re.sub(r'\d+', '', metin)
    return re.sub(r'\s+', ' ', metin).strip()

df['text_clean'] = df['text'].apply(temizle)

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text_clean'])
X = pad_sequences(tokenizer.texts_to_sequences(df['text_clean']),
                  maxlen=MAX_LEN, padding='post', truncating='post')
y = df['label'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
print(f"✅ Eğitim:{X_train.shape[0]} | Val:{X_val.shape[0]} | Test:{X_test.shape[0]}")

# ── ATTENTION KATMANI ────────────────────────────────────────
class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                  initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, x):
        score  = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        weight = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * weight, axis=1)

# ════════════════════════════════════════════════════════════
# GELİŞTİRİLMİŞ MODEL 1: Custom MLP v2
# Değişiklikler:
#   ✗ BatchNormalization kaldırıldı (eğitim kararsızlığı yapıyordu)
#   ✓ L2 regularization eklendi (overfitting azaltır)
#   ✓ LeakyReLU kullanıldı (ölü nöron sorununu önler)
#   ✓ Daha güçlü Dropout (0.5 → 0.4 → 0.3)
#   ✓ GlobalAvg + GlobalMax concat korundu (iyi çalışıyordu)
# ════════════════════════════════════════════════════════════
def build_mlp_v2():
    L2 = 1e-4
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(MAX_WORDS, EMBED_DIM)(inp)

    avg = layers.GlobalAveragePooling1D()(x)
    mx  = layers.GlobalMaxPooling1D()(x)
    x   = layers.Concatenate()([avg, mx])          # 256 boyut

    x   = layers.Dense(256, kernel_regularizer=regularizers.l2(L2))(x)
    x   = layers.LeakyReLU(negative_slope=0.1)(x)
    x   = layers.Dropout(0.5)(x)

    x   = layers.Dense(128, kernel_regularizer=regularizers.l2(L2))(x)
    x   = layers.LeakyReLU(negative_slope=0.1)(x)
    x   = layers.Dropout(0.4)(x)

    x   = layers.Dense(64, kernel_regularizer=regularizers.l2(L2))(x)
    x   = layers.LeakyReLU(negative_slope=0.1)(x)
    x   = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inp, out, name='Custom_MLP_v2')
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ════════════════════════════════════════════════════════════
# GELİŞTİRİLMİŞ MODEL 2: Attention-BiLSTM
# Değişiklikler:
#   ✓ Attention katmanı eklendi (önemli kelimelere odaklanır)
#   ✓ recurrent_dropout eklendi (LSTM içi regularization)
#   ✓ SpatialDropout1D korundu
# ════════════════════════════════════════════════════════════
def build_bilstm_v2():
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(MAX_WORDS, EMBED_DIM)(inp)
    x   = layers.SpatialDropout1D(0.3)(x)

    x   = layers.Bidirectional(
              layers.LSTM(128, return_sequences=True,
                          dropout=0.2))(x)
    x   = AttentionLayer()(x)          # ← dikkat katmanı

    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inp, out, name='Attention_BiLSTM_v2')
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ════════════════════════════════════════════════════════════
# GELİŞTİRİLMİŞ MODEL 3: TextCNN v2
# Değişiklikler:
#   ✓ 4 farklı kernel boyutu: 2,3,4,5 (önceki: 3,4,5)
#   ✓ Her kernel için 128 filtre (aynı)
#   ✓ L2 regularization Conv katmanlarına eklendi
#   ✓ GlobalMaxPool + GlobalAvgPool birleştirildi (her kernel için)
# ════════════════════════════════════════════════════════════
def build_textcnn_v2():
    L2  = 1e-4
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(MAX_WORDS, EMBED_DIM)(inp)

    convs = []
    for ks in [2, 3, 4, 5]:          # 4 farklı n-gram boyutu
        c    = layers.Conv1D(128, kernel_size=ks, activation='relu',
                             padding='same',
                             kernel_regularizer=regularizers.l2(L2))(x)
        cmax = layers.GlobalMaxPooling1D()(c)
        cavg = layers.GlobalAveragePooling1D()(c)
        convs.append(layers.Concatenate()([cmax, cavg]))   # 256 / kernel

    x   = layers.Concatenate()(convs)     # 4 × 256 = 1024
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inp, out, name='TextCNN_v2')
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ── CALLBACK ─────────────────────────────────────────────────
def get_callbacks(name):
    return [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=2, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(f'best_{name}.keras',
                                         monitor='val_accuracy',
                                         save_best_only=True, verbose=0),
    ]

fit_kw = dict(epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val))

# ── EĞİTİM ───────────────────────────────────────────────────
print("\n" + "="*55)
print("=== Geliştirilmiş Model 1: Custom MLP v2 ===")
mlp_v2   = build_mlp_v2()
mlp_v2.summary()
mlp_v2_h = mlp_v2.fit(X_train, y_train,
                        callbacks=get_callbacks('mlp_v2'), **fit_kw)

print("\n" + "="*55)
print("=== Geliştirilmiş Model 2: Attention-BiLSTM ===")
bilstm_v2   = build_bilstm_v2()
bilstm_v2.summary()
bilstm_v2_h = bilstm_v2.fit(X_train, y_train,
                              callbacks=get_callbacks('bilstm_v2'), **fit_kw)

print("\n" + "="*55)
print("=== Geliştirilmiş Model 3: TextCNN v2 ===")
cnn_v2   = build_textcnn_v2()
cnn_v2.summary()
cnn_v2_h = cnn_v2.fit(X_train, y_train,
                        callbacks=get_callbacks('cnn_v2'), **fit_kw)

# ── DEĞERLENDİRME ────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

def evaluate(model, name):
    y_pred = (model.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    print(f"\n{'─'*45}\n  {name}")
    print(classification_report(y_test, y_pred,
                                  target_names=['Negatif','Pozitif']))
    return {
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_test, y_pred),  4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred),    4),
        'F1-Score':  round(f1_score(y_test, y_pred),        4),
        'CM':        confusion_matrix(y_test, y_pred),
    }

print("\n\n" + "="*55)
print("TEST SETİ — GELİŞTİRİLMİŞ MODELLER")
r_mlp  = evaluate(mlp_v2,    'Custom MLP v2')
r_bil  = evaluate(bilstm_v2, 'Attention-BiLSTM')
r_cnn  = evaluate(cnn_v2,    'TextCNN v2')

# Baseline vs Geliştirilmiş karşılaştırma
print("\n\n" + "="*55)
print("📊 BASELINE vs GELİŞTİRİLMİŞ KARŞILAŞTIRMA")
compare = pd.DataFrame([
    {'Model': 'Custom MLP (baseline)',      'Accuracy': 0.8741, 'F1': 0.8791},
    {'Model': 'Custom MLP v2 ✨',           'Accuracy': r_mlp['Accuracy'], 'F1': r_mlp['F1-Score']},
    {'Model': 'BiLSTM (baseline)',          'Accuracy': 0.8906, 'F1': 0.8913},
    {'Model': 'Attention-BiLSTM ✨',        'Accuracy': r_bil['Accuracy'], 'F1': r_bil['F1-Score']},
    {'Model': 'TextCNN (baseline)',         'Accuracy': 0.8896, 'F1': 0.8912},
    {'Model': 'TextCNN v2 ✨',              'Accuracy': r_cnn['Accuracy'], 'F1': r_cnn['F1-Score']},
])
print(compare.to_string(index=False))

# ── GRAFİKLER ────────────────────────────────────────────────
os.makedirs('grafik', exist_ok=True)

# 1. Eğitim eğrileri
COLORS = ['#e74c3c', '#3498db', '#27ae60']
NAMES  = ['Custom MLP v2', 'Attention-BiLSTM', 'TextCNN v2']
HISTS  = [mlp_v2_h, bilstm_v2_h, cnn_v2_h]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for hist, name, color in zip(HISTS, NAMES, COLORS):
    axes[0].plot(hist.history['loss'],         color=color, lw=2, label=f'{name} Train')
    axes[0].plot(hist.history['val_loss'],     color=color, lw=2, ls='--', label=f'{name} Val')
    axes[1].plot(hist.history['accuracy'],     color=color, lw=2, label=f'{name} Train')
    axes[1].plot(hist.history['val_accuracy'], color=color, lw=2, ls='--', label=f'{name} Val')

axes[0].set_title('Geliştirilmiş — Eğitim Kaybı', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_title('Geliştirilmiş — Doğruluk', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('grafik/gelistirilmis_egitim.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ grafik/gelistirilmis_egitim.png")

# 2. Confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, r, name, cmap in zip(axes,
                               [r_mlp, r_bil, r_cnn],
                               NAMES,
                               ['Reds','Blues','Greens']):
    sns.heatmap(r['CM'], annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Negatif','Pozitif'],
                yticklabels=['Negatif','Pozitif'], annot_kws={'size':13})
    ax.set_title(f'{name}\nAcc:{r["Accuracy"]} | F1:{r["F1-Score"]}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gerçek')
plt.suptitle('Geliştirilmiş Modeller — Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafik/gelistirilmis_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ grafik/gelistirilmis_confusion.png")

# 3. Baseline vs Geliştirilmiş bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
model_labels = ['Custom\nMLP', 'BiLSTM', 'TextCNN']
baseline_acc = [0.8741, 0.8906, 0.8896]
improved_acc = [r_mlp['Accuracy'], r_bil['Accuracy'], r_cnn['Accuracy']]
baseline_f1  = [0.8791, 0.8913, 0.8912]
improved_f1  = [r_mlp['F1-Score'], r_bil['F1-Score'], r_cnn['F1-Score']]

x     = np.arange(len(model_labels))
width = 0.35

for ax, base, impr, title, metric in zip(
    axes,
    [baseline_acc, baseline_f1],
    [improved_acc, improved_f1],
    ['Accuracy Karşılaştırması', 'F1-Score Karşılaştırması'],
    ['Accuracy', 'F1-Score']
):
    b1 = ax.bar(x - width/2, base, width, label='Baseline',
                color='#95a5a6', edgecolor='black', alpha=0.85)
    b2 = ax.bar(x + width/2, impr, width, label='Geliştirilmiş',
                color=['#e74c3c','#3498db','#27ae60'], edgecolor='black', alpha=0.85)
    for bar, val in list(zip(b1, base)) + list(zip(b2, impr)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(model_labels)
    ax.set_ylim(0.83, 0.97); ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel(metric)

plt.suptitle('Baseline vs Geliştirilmiş Model Karşılaştırması',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafik/baseline_vs_gelistirilmis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ grafik/baseline_vs_gelistirilmis.png")

# ── MODELLERİ KAYDET ────────────────────────────────────────
mlp_v2.save('model_mlp_v2.keras')
bilstm_v2.save('model_attn_bilstm.keras')
cnn_v2.save('model_cnn_v2.keras')

import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\n✅ Tüm geliştirilmiş modeller kaydedildi!")
print("\n📊 ÖZET:")
print(f"  Custom MLP:   {0.8741} → {r_mlp['Accuracy']}  (F1: {0.8791} → {r_mlp['F1-Score']})")
print(f"  BiLSTM:       {0.8906} → {r_bil['Accuracy']}  (F1: {0.8913} → {r_bil['F1-Score']})")
print(f"  TextCNN:      {0.8896} → {r_cnn['Accuracy']}  (F1: {0.8912} → {r_cnn['F1-Score']})")
