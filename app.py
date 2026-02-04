import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# --------------------
# SETTINGS
# --------------------

MODEL_PATH = "models/cnn_audio_model.h5"
IMG_SIZE = 64
SAMPLE_RATE = 22050

CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

plt.switch_backend("Agg")

# --------------------
# LOAD MODEL
# --------------------

@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_cnn()

# --------------------
# AUDIO → SPECTROGRAM
# --------------------

def audio_to_spectrogram(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=1.0   # FAST realtime window
    )

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=512
    )

    spec_db = librosa.power_to_db(spec, ref=np.max)

    # Convert to image in memory
    fig, ax = plt.subplots(figsize=(3,3))
    librosa.display.specshow(spec_db, sr=sr, ax=ax)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0

    return img_array, spec_db, audio, sr

# --------------------
# VISUALIZATION
# --------------------

def plot_waveform(audio, sr):
    fig, ax = plt.subplots()
    ax.plot(audio)
    ax.set_title("Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_spectrogram(spec_db, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        spec_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

# --------------------
# PREDICTION
# --------------------

def predict_audio(file_path):
    img, spec_db, audio, sr = audio_to_spectrogram(file_path)

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    return preds, spec_db, audio, sr

# --------------------
# STREAMLIT UI
# --------------------

st.set_page_config(page_title="Real-Time Audio CNN", layout="centered")

st.title("🎧 Real-Time Audio Classification (CNN)")
st.write("Upload an audio file to classify sound using deep learning")

show_viz = st.checkbox("Show waveform & spectrogram", value=True)

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    with st.spinner("Processing..."):
        preds, spec_db, audio, sr = predict_audio(uploaded_file)

    predicted_class = CLASSES[np.argmax(preds)]
    confidence = np.max(preds)

    st.success(f"🎯 Prediction: {predicted_class} ({confidence:.2f})")

    # Probability bars
    st.subheader("Class Probabilities")

    for cls, prob in zip(CLASSES, preds):
        st.write(f"{cls}: {prob:.2f}")
        st.progress(float(prob))

    if show_viz:
        st.subheader("Audio Visualization")

        plot_waveform(audio, sr)
        plot_spectrogram(spec_db, sr)