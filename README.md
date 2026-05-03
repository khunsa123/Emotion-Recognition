# 🎙️ Speech Emotion Recognition Using Deep Learning

> Deep learning pipeline for multi-class speech emotion classification across four public corpora.  
> Related publication: **Ahmed, W., Riaz, S., Iftikhar, K., Konur, S. (2023)**. *Speech Emotion Recognition Using Deep Learning.* Springer LNCS Vol. 14381, SGAI 2023.

---

## 📌 Overview

This repository contains the code and experiments for a CNN-based speech emotion recognition system trained and evaluated across four independent speech emotion corpora. The work investigates the generalisation of deep learning models across heterogeneous acoustic datasets — a core challenge in robust real-world speech emotion recognition.

**Core research question:** Can a single deep learning architecture trained on multiple speech corpora generalise robustly across speakers, recording conditions, and emotional label schemes?

---

## 📂 Repository Structure

```
Emotion-Recognition/
│
├── Speech Emotion Recognition/
│   ├── feature_extraction.ipynb       # MFCC, ZCR, RMS, spectral feature extraction
│   ├── data_augmentation.ipynb        # Noise injection, pitch shift, time-stretch
│   ├── model_training.ipynb           # CNN architecture, training loop, cross-validation
│   ├── evaluation.ipynb               # Accuracy, confusion matrix, per-class metrics
│   └── cross_corpus_analysis.ipynb    # Cross-corpus generalisation evaluation
│
├── figures/                           # Visualisations and result plots
└── README.md
```

---

## 🗂️ Datasets

Four publicly available speech emotion corpora were used:

| Dataset | Language | Speakers | Emotions | Size |
|---------|----------|----------|----------|------|
| [RAVDESS](https://zenodo.org/record/1188976) | English | 24 | 8 | 2,452 |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) | English | 2 | 7 | 2,800 |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | English | 91 | 6 | 7,442 |
| [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/) | English | 4 | 7 | 480 |

> **Note:** Datasets are not included in this repository due to licensing. Download links are provided above. Place datasets in a `data/` directory following the structure described in `feature_extraction.ipynb`.

---

## 🧠 Methodology

### 1. Feature Extraction
The following acoustic features were extracted from each audio file:

- **MFCC** (Mel-Frequency Cepstral Coefficients) — captures spectral envelope of speech
- **Zero-Crossing Rate (ZCR)** — measures signal frequency characteristics
- **RMS Energy** — captures loudness and intensity variation
- **Spectral Centroid** — indicates brightness of sound
- **Spectral Rolloff** — captures frequency distribution shape
- **Spectral Bandwidth** — measures spread of spectral energy

### 2. Data Augmentation
To improve generalisation and address class imbalance:
- Gaussian noise injection
- Pitch shifting (±2 semitones)
- Time stretching (0.8× and 1.2× speed)

### 3. Model Architecture
A Convolutional Neural Network (CNN) trained on the concatenated feature matrix:
- Input: concatenated feature vector per audio segment
- Architecture: Conv1D → BatchNorm → MaxPool → Dropout → Dense
- Output: softmax classification over emotion classes

### 4. Evaluation
- **k-fold cross-validation** (k=5) within each corpus
- **Cross-corpus evaluation** — training on combined corpora, testing on held-out dataset
- Metrics: accuracy, weighted F1-score, per-class precision/recall, confusion matrix

---

## 📊 Results

| Evaluation Setting | Accuracy |
|--------------------|----------|
| Within-corpus (average across datasets) | ~76% |
| Cross-corpus generalisation | See notebook |

> Full results, confusion matrices, and learning curves are available in `evaluation.ipynb`.

---

## 🛠️ Requirements

```bash
pip install numpy pandas librosa scikit-learn tensorflow matplotlib seaborn
```

**Python:** 3.8+  
**Key libraries:** `librosa` (audio processing), `scikit-learn` (cross-validation), `TensorFlow/Keras` (model training)

---

## 🚀 Quick Start

```python
# 1. Clone the repository
git clone https://github.com/khunsa123/Emotion-Recognition.git
cd Emotion-Recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets and place in data/ directory

# 4. Run feature extraction
jupyter notebook "Speech Emotion Recognition/feature_extraction.ipynb"

# 5. Train model
jupyter notebook "Speech Emotion Recognition/model_training.ipynb"
```

---
## 📄 Publication

This repository contains the open-source implementation accompanying the following peer-reviewed publication:

> Ahmed, W., Riaz, S., Iftikhar, K., & Konur, S. (2023). **Speech Emotion Recognition Using Deep Learning.** In M. Bramer & F. Stahl (Eds.), *Artificial Intelligence XL (SGAI 2023)*, Lecture Notes in Computer Science, Vol. 14381, pp. 191–197. Springer Nature Switzerland.  
> 🔗 [doi.org/10.1007/978-3-031-47994-6_14](https://doi.org/10.1007/978-3-031-47994-6_14)

The full paper is available via Springer at the link above. This repository shares the implementation code only; the paper text, figures, and tables remain under Springer Nature copyright.

If you find this work useful, please cite:

```bibtex
@InProceedings{10.1007/978-3-031-47994-6_14,
  author    = {Ahmed, Waqar and Riaz, Sana and Iftikhar, Khunsa and Konur, Savas},
  editor    = {Bramer, Max and Stahl, Frederic},
  title     = {Speech Emotion Recognition Using Deep Learning},
  booktitle = {Artificial Intelligence XL},
  year      = {2023},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {191--197},
  isbn      = {978-3-031-47994-6},
  doi       = {10.1007/978-3-031-47994-6_14}
}
```

---

## 👩‍🔬 Author

**Khunsa Iftikhar**  
Computational Neuroscience & AI Researcher  
🔗 [Google Scholar](https://scholar.google.com/citations?hl=en&user=Q-mM508AAAAJ) | [LinkedIn](https://www.linkedin.com/in/khunsa-iftikhar/) | [Website](https://sites.google.com/view/khunsa-iftikhar/)

---

## 📬 Contact

For questions or collaboration inquiries: **khunsaiftikhar123@gmail.com**
