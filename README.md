# 🎧 Audio Classification System (CNN + Streamlit)

A deep learning–based real-time audio classification web application that converts audio signals into spectrograms and uses a Convolutional Neural Network (CNN) to classify sounds.

The app supports live microphone input, audio upload, spectrogram visualization, and real-time prediction via an interactive Streamlit UI.

---

## 🚀 Live Demo

https://audio-classification-system.streamlit.app/

---

## 📌 Features

✅ Real-time microphone recording  
✅ Audio file upload support  
✅ Waveform visualization  
✅ Spectrogram generation  
✅ CNN-based deep learning prediction  
✅ Fast inference optimized for real-time use  
✅ Clean Streamlit web interface  

---

## 🧠 Model Architecture

Workflow:

Audio Input  
↓  
Spectrogram Generation (Librosa)  
↓  
CNN Model (TensorFlow/Keras)  
↓  
Predicted Audio Class  

---

## 🛠️ Tech Stack

Language: Python 3.10  
Deep Learning: TensorFlow / Keras  
Audio Processing: Librosa, SoundFile  
Visualization: Matplotlib  
Web UI: Streamlit  
Deployment: Streamlit Cloud  
Version Control: Git + GitHub  

---

## 📂 Project Structure

audio-classification-system/

├── app.py  
├── requirements.txt  
├── README.md  

├── models/  
│   └── cnn_audio_model.h5  

├── src/  
│   ├── cnn_model.py  
│   ├── train_cnn.py  
│   └── generate_specs.py  

└── data/  

---

## ⚙️ Installation

1. Clone repository

git clone https://github.com/YOUR_USERNAME/audio-classification-system.git  
cd audio-classification-system  

2. Create virtual environment

python -m venv venv  

Windows:
venv\Scripts\activate  

Mac/Linux:
source venv/bin/activate  

3. Install dependencies

pip install -r requirements.txt  

---

## ▶️ Run Application

streamlit run app.py  

Open browser:

http://localhost:8501  

---

## 🏋️ Train CNN Model (Optional)

Generate spectrogram dataset:

python src/generate_specs.py  

Train CNN:

python src/train_cnn.py  

Model saved to:

models/cnn_audio_model.h5  

---

## 🌐 Deployment

Using Streamlit Cloud:

1. Push to GitHub  
2. Go to https://share.streamlit.io  
3. Select repository  
4. Set main file: app.py  
5. Click Deploy  

---

## 📈 Performance Optimizations

✔ Cached model loading  
✔ Fast spectrogram computation  
✔ Optimized inference pipeline  

---

## 👨‍💻 Author

Mohit Sharma  

GitHub: https://github.com/lonwulf07/

---

## ⭐ Support

If you like this project, give it a star ⭐