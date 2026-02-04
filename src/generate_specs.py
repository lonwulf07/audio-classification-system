import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths (adjust if needed)
BASE_PATH = "data/raw/UrbanSound8K"
AUDIO_PATH = os.path.join(BASE_PATH, "audio")
CSV_PATH = os.path.join(BASE_PATH, "metadata", "UrbanSound8K.csv")

SAVE_PATH = "data/spectrograms"

os.makedirs(SAVE_PATH, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():

    filename = row["slice_file_name"]
    fold = f"fold{row['fold']}"
    label = row["class"]

    audio_file = os.path.join(AUDIO_PATH, fold, filename)

    class_dir = os.path.join(SAVE_PATH, label)
    os.makedirs(class_dir, exist_ok=True)

    try:
        audio, sr = librosa.load(audio_file, sr=22050, duration=1.0)

        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )

        spec_db = librosa.power_to_db(spec, ref=np.max)

        plt.figure(figsize=(2,2))
        librosa.display.specshow(spec_db)
        plt.axis("off")

        save_file = os.path.join(
            class_dir,
            filename.replace(".wav", ".png")
        )

        plt.savefig(save_file, bbox_inches="tight", pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Skipping {audio_file}: {e}")

print("✅ UrbanSound8K spectrograms created successfully")