# 🎙️ Speech Emotion Recognition Using Deep Learning

> **Published:** Ahmed, W., Riaz, S., Iftikhar, K., Konur, S. (2023). *Speech Emotion Recognition Using Deep Learning.* Springer LNCS Vol. 14381, SGAI 2023.
> [Read on ResearchGate](https://www.researchgate.net/publication/375476014_Speech_Emotion_Recognition_Using_Deep_Learning)

---

## 📌 Project Overview

This project presents a **CNN-based deep learning pipeline for Speech Emotion Recognition (SER)** trained and evaluated across four public emotion speech corpora. The system extracts handcrafted audio features from raw speech signals and classifies them into discrete emotional categories, with a graphical user interface (GUI) built for demonstration and real-world testing.

---

## 🎯 Objectives

- Develop a robust SER pipeline using CNNs on multi-corpus speech data
- Extract and evaluate discriminative audio features across different datasets
- Apply data augmentation and cross-validation to improve cross-corpus generalisation
- Build a GUI-based demo for real-time emotion classification from audio input

---

## 🧠 Methods & Techniques

- Audio preprocessing and feature extraction using Librosa
- **Feature set:** MFCCs, Zero Crossing Rate (ZCR), RMS Energy, Spectral Rolloff, Chroma features
- **Data augmentation:** noise injection, time stretching, and pitch shifting to improve model robustness
- CNN-based classification with dropout regularisation
- k-fold cross-validation for reliable performance estimation
- GUI implementation for demonstration and real-time testing

### 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score (per emotion class)
- Confusion Matrix

---

## 🤖 Model Implemented

- **Convolutional Neural Network (CNN)** — 1D/2D feature maps over extracted audio features
- Compared against baseline classical ML models

---

## 📊 Datasets

Four publicly available, multi-speaker, multi-emotion speech corpora:

| Dataset | Description |
|---------|-------------|
| **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song |
| **TESS** | Toronto Emotional Speech Set |
| **CREMA-D** | Crowd-sourced Emotional Multimodal Actors Dataset |
| **SAVEE** | Surrey Audio-Visual Expressed Emotion |

- Combined multi-corpus training to improve generalisation across speakers and recording conditions

---

## 📈 Results

- Achieved **~76% classification accuracy** across emotion categories
- Data augmentation improved model stability on underrepresented emotion classes
- CNN outperformed classical ML baselines on multi-corpus evaluation
- Experiments fully documented for reproducibility

---

## 🔬 Research Significance

This work was **published at SGAI 2023 (Springer LNCS)**, contributing to the field of affective computing by:

- Demonstrating effective multi-corpus training for improved SER generalisation
- Providing a systematic comparison of audio feature sets for emotion classification
- Delivering a reproducible, end-to-end deep learning pipeline with a functional GUI
- Laying the groundwork for multimodal extensions incorporating EEG and physiological signals

---

## 🛠️ Tech Stack

- **Programming:** Python
- **DL Framework:** TensorFlow / Keras
- **Audio Processing:** Librosa, SciPy
- **ML:** scikit-learn
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, Google Colab

---

## 📂 Project Structure

```
Speech Emotion Recognition/
│── feature_extraction.py       # MFCC, ZCR, RMS, spectral feature extraction
│── data_augmentation.py        # Noise injection, time stretch, pitch shift
│── model.py                    # CNN architecture and training pipeline
│── evaluate.py                 # Cross-validation, metrics, confusion matrix
│── gui.py                      # GUI for real-time emotion classification
│── notebooks/                  # Exploratory analysis and visualisations
└── README.md
```

---

## 🚀 Future Work

- Extension to EEG-based emotion recognition using BCI devices
- Multimodal fusion of speech and EEG signals for richer affective state modelling
- Transformer-based audio models (e.g., wav2vec 2.0) for end-to-end SER
- Cross-lingual and cross-cultural emotion generalisation studies

---

## 📬 Contact

**Khunsa Iftikhar**
📧 [khunsaiftikhar123@gmail.com](mailto:khunsaiftikhar123@gmail.com)
🔗 [linkedin.com/in/khunsa-iftikhar](https://www.linkedin.com/in/khunsa-iftikhar/)
