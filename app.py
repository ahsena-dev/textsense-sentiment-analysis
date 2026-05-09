# ============================================================
# BIM 430 — SentiMind FastAPI Backend
# Kurulum: pip install fastapi uvicorn
# Çalıştır: uvicorn app:app --reload
# Aç: http://localhost:8000
# ============================================================

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import re, pickle

# ── ATTENTION KATMANI (model yüklemek için gerekli) ──────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

# ── UYGULAMA ─────────────────────────────────────────────────
app = FastAPI(title="SentiMind API")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── MODEL YÜKLE ──────────────────────────────────────────────
print("📦 Modeller yükleniyor...")
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 150

mlp_model    = keras.models.load_model('model_mlp_v2.keras')
bilstm_model = keras.models.load_model('model_attn_bilstm.keras',
                                        custom_objects={'AttentionLayer': AttentionLayer})
cnn_model    = keras.models.load_model('model_cnn_v2.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print("✅ 3 model hazır — sunucu başlatılıyor...")

# ── METİN ÖN İŞLEME ─────────────────────────────────────────
def temizle(metin: str) -> str:
    metin = str(metin).lower()
    metin = re.sub(r'http\S+|www\S+', '', metin)
    metin = re.sub(r'[^\w\s]', ' ', metin)
    metin = re.sub(r'\d+', '', metin)
    return re.sub(r'\s+', ' ', metin).strip()

def hazirla(text: str):
    seq = tokenizer.texts_to_sequences([temizle(text)])
    return pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

# ── TAHMİN ───────────────────────────────────────────────────
def tahmin_et(text: str) -> dict:
    padded   = hazirla(text)
    mlp_p    = float(mlp_model.predict(padded,    verbose=0)[0][0])
    bilstm_p = float(bilstm_model.predict(padded, verbose=0)[0][0])
    cnn_p    = float(cnn_model.predict(padded,    verbose=0)[0][0])

    # Ağırlıklı ortalama — TextCNN v2 en iyi model
    avg    = mlp_p * 0.30 + bilstm_p * 0.30 + cnn_p * 0.40
    is_pos = avg >= 0.5
    conf   = round(avg * 100) if is_pos else round((1 - avg) * 100)

    # 1–5 yıldız
    if is_pos:
        stars = 5 if avg > 0.90 else 4 if avg > 0.75 else 3
    else:
        stars = 1 if avg < 0.10 else 2

    descs = {1:"Çok olumsuz", 2:"Olumsuz", 3:"Olumlu", 4:"Çok olumlu", 5:"Mükemmel"}

    return {
        "text":             text,
        "isPositive":       is_pos,
        "label":            "Pozitif" if is_pos else "Negatif",
        "confidence":       conf,
        "average_score":    round(avg, 3),
        "stars":            stars,
        "star_label":       f"{stars} Yıldız",
        "star_description": descs[stars],
        "models": {
            "mlp":  round(mlp_p * 100),
            "lstm": round(bilstm_p * 100),
            "cnn":  round(cnn_p * 100),
        },
        "model_labels": {
            "mlp":  "Pozitif" if mlp_p  >= 0.5 else "Negatif",
            "lstm": "Pozitif" if bilstm_p >= 0.5 else "Negatif",
            "cnn":  "Pozitif" if cnn_p  >= 0.5 else "Negatif",
        }
    }

# ── ROUTE'LAR ────────────────────────────────────────────────
class TextReq(BaseModel):
    text: str

class BulkReq(BaseModel):
    texts: List[str]

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/analyze")
def analyze(req: TextReq):
    return tahmin_et(req.text)

@app.post("/analyze_bulk")
def analyze_bulk(req: BulkReq):
    return [tahmin_et(t) for t in req.texts[:200]]

@app.get("/health")
def health():
    return {"status": "ok", "models": ["MLP v2", "Attention-BiLSTM", "TextCNN v2"]}
