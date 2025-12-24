import streamlit as st
import tensorflow as tf
import torch
import pickle
import numpy as np
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# Matikan wandb
os.environ["WANDB_DISABLED"] = "true"

# --- DOWNLOAD NLTK ---
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

download_nltk_resources()

# --- CONFIG HALAMAN ---
st.set_page_config(page_title="Suicide Detection App", layout="wide")

# --- PREPROCESSING (HANYA UNTUK LSTM) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_for_lstm(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- LOAD MODELS ---
@st.cache_resource
def load_all_models():
    # Load Label Encoder
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Cari tahu mana index untuk 'suicide' secara dinamis
    # Biasanya label encoder: 0 = non-suicide, 1 = suicide (tergantung alfabet)
    labels_list = le.classes_.tolist()
    suicide_idx = labels_list.index('suicide')
    
    # Load LSTM
    model_lstm = tf.keras.models.load_model('models/model_lstm.h5', compile=False)
    with open('models/tokenizer_lstm.pkl', 'rb') as f:
        tokenizer_lstm = pickle.load(f)
        
    # Load Transformers (Gunakan teks asli, bukan cleaned)
    pipe_db = pipeline("text-classification", model="models/model_distilbert", device=-1, top_k=None)
    pipe_rb = pipeline("text-classification", model="models/model_roberta", device=-1, top_k=None)
    
    return le, suicide_idx, model_lstm, tokenizer_lstm, pipe_db, pipe_rb

# --- FUNGSI PEMBANTU DISPLAY ---
def display_result(col, title, prob_suicide):
    # Tentukan Label berdasarkan threshold 50%
    label = "SUICIDE" if prob_suicide > 50 else "NON-SUICIDE"
    prob_non = 100 - prob_suicide
    
    col.subheader(title)
    # Tampilkan Label dengan warna
    if label == "SUICIDE":
        col.error(f"**{label}**")
    else:
        col.success(f"**{label}**")
    
    col.write(f"**Suicide:** {prob_suicide:.2f}%")
    col.progress(min(prob_suicide / 100, 1.0))
    
    col.write(f"**Non-Suicide:** {prob_non:.2f}%")
    col.progress(min(prob_non / 100, 1.0))

# --- MAIN APP ---
try:
    le, suicide_idx, model_lstm, tokenizer_lstm, pipe_db, pipe_rb = load_all_models()
except Exception as e:
    st.error(f"Gagal memuat model. Detail: {e}")
    st.stop()

st.title("🛡️ Suicide Detection Multi-Model System")
st.markdown("Sistem ini mendeteksi indikasi bunuh diri menggunakan tiga arsitektur AI berbeda.")
st.divider()

user_input = st.text_area("Masukkan teks postingan (English):", 
                          placeholder="Contoh: I feel very lonely and I don't want to be here anymore...",
                          height=150)

if st.button("Jalankan Deteksi"):
    if user_input.strip():
        # Preprocessing HANYA untuk LSTM
        cleaned_lstm = clean_text_for_lstm(user_input)
        
        with st.spinner('Sedang Menganalisis...'):
            # --- 1. Prediksi LSTM (Menggunakan Cleaned Text) ---
            seq = tokenizer_lstm.texts_to_sequences([cleaned_lstm])
            padded = pad_sequences(seq, maxlen=150)
            res_lstm_raw = model_lstm.predict(padded, verbose=0)[0][0]
            prob_suicide_lstm = float(res_lstm_raw) * 100
            
            # --- 2. Prediksi DistilBERT (Menggunakan Raw Text) ---
            # Cari skor yang sesuai dengan index 'suicide' di label encoder
            out_db = pipe_db(user_input)[0]
            prob_suicide_db = next(item['score'] for item in out_db if str(suicide_idx) in item['label']) * 100

            # --- 3. Prediksi RoBERTa (Menggunakan Raw Text) ---
            out_rb = pipe_rb(user_input)[0]
            prob_suicide_rb = next(item['score'] for item in out_rb if str(suicide_idx) in item['label']) * 100

        # Tampilan Hasil dalam 3 Kolom
        col1, col2, col3 = st.columns(3)
        
        display_result(col1, "LSTM", prob_suicide_lstm)
        display_result(col2, "DistilBERT", prob_suicide_db)
        display_result(col3, "RoBERTa", prob_suicide_rb)
        
    else:
        st.warning("Masukkan teks terlebih dahulu!")

st.divider()
st.caption("UAP Pembelajaran Mesin 2025 - Gunakan hasil ini hanya untuk tujuan edukasi.")