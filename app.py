import io
import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import librosa
from pydub import AudioSegment


# -------------------- CONFIG / THEME --------------------
st.set_page_config(page_title="VoiceScope", page_icon="🎙️", layout="centered")
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
      background-image: url('https://img.freepik.com/free-photo/abstract-digital-grid-black-background_53876-97647.jpg');
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
    }
    .main-title { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; text-align:center; color: #fff; text-shadow: 1px 1px 7px #000; }
    .sub { color: #d1d5db; margin-bottom: 2rem; text-align:center; font-weight:500; font-size:1.2rem; text-shadow: 1px 1px 4px #000; }
    .pred-card { border: 1px solid #e5e7eb; border-radius:15px; padding:20px; background:rgba(255,255,255,0.88); box-shadow:0 8px 16px rgba(0,0,0,0.15); max-width:320px; margin:24px auto; font-weight:600; text-align:center; color:#111827; }
    </style>
""", unsafe_allow_html=True)


# -------------------- CONSTANTS --------------------
TIME_STEPS = 120
NUM_FEATURES = 40  # must match your model input

GENDER_LABELS = [
    ("female"),
    ("male")
]

AGE_LABELS = {
    0: "twenties",
    1: "thirties",
    2: "fifties",
    3: "sixties"
}


# -------------------- HELPERS --------------------
@st.cache_resource(show_spinner=False)
def robust_load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        return None


def mp3_bytes_to_wav_ndarray(file_bytes: bytes, target_sr=16000):
    try:
        mp3 = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
        buf = io.BytesIO()
        mp3.set_channels(1).set_frame_rate(target_sr).export(buf, format="wav")
        buf.seek(0)
        y, sr = librosa.load(buf, sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        raise e


def extract_mfcc_tensor(y, sr, num_mfcc=NUM_FEATURES, time_steps=TIME_STEPS):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc).T
    if mfcc.shape[0] < time_steps:
        pad_width = time_steps - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:time_steps]
    return mfcc[np.newaxis, ...].astype(np.float32)


def predict_combined_label(single_model, wav_bytes, filename=None, return_probs=True):
    y, sr = mp3_bytes_to_wav_ndarray(wav_bytes, target_sr=16000)
    x = extract_mfcc_tensor(y, sr)
    preds = single_model.predict(x)[0]
    class_idx = int(np.argmax(preds))
    gender, age_original = GENDER_LABELS[class_idx],AGE_LABELS[class_idx]

    predicted_age_label = AGE_LABELS.get(class_idx % 4, "Unknown")

    if return_probs:
        return predicted_age_label, float(preds[class_idx]), gender, float(preds[class_idx])
    else:
        return predicted_age_label, gender


# -------------------- SIDEBAR --------------------
st.sidebar.header("Settings ")
model_path = st.sidebar.text_input("Model path (.keras/.h5)", "my_model.h5")


# -------------------- APP --------------------
st.markdown('<div class="main-title">🎙️ Voice Age & Gender Classifier (Combined)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload an MP3 clip and get instant combined age and gender predictions.</div>', unsafe_allow_html=True)


uploaded = st.file_uploader("Drop an .mp3 file here", type=["mp3"])

if uploaded:
    audio_bytes = uploaded.read()
    st.audio(io.BytesIO(audio_bytes), format="audio/mpeg")

    with st.spinner("Loading model..."):
        model = robust_load_model(model_path)

    if model is not None:
        with st.spinner("Predicting..."):
            try:
                age_label, age_conf, gender_label, gender_conf = predict_combined_label(model, audio_bytes, filename=uploaded.name, return_probs=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Age Prediction")
                    st.markdown(f'<div class="pred-card">Predicted: <b>{age_label}</b><br/>Confidence: {age_conf:.2%}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("#### Gender Prediction")
                    st.markdown(f'<div class="pred-card">Predicted: <b>{gender_label}</b><br/>Confidence: {gender_conf:.2%}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to process audio: {e}")
    else:
        st.error("Model not loaded. Please check the model file path and format.")

st.caption("Tip: Upload short clips (≤10 seconds) for best results. Model is cached for speed.")
