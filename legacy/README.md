# 🎵 Multimodal Music Discovery Engine
**NYU CSCI-UA 473: Fundamentals of Machine Learning (Final Project)**

---

## 🔗 Live Demo & Documentation
*   **[🟢 Live Web App (Streamlit)](https://youngsang-cho1-ml-final-appstreamlit-app-pgt8wj.streamlit.app/)**
*   **[📄 Mathematical Design Document (DESIGN_DOC.md)](./DESIGN_DOC.md)**: Deep dive into the from-scratch algorithms (PCA, K-Means++, Hybrid Metrics).

---

## 📖 Project Objective
A high-fidelity music recommendation engine that focuses on **Acoustic Consistency**. We deliberately avoid black-box libraries, implementing core ML algorithms (Manual PCA, Manual K-Means++) from scratch in `NumPy` to demonstrate a deep understanding of vector space models and multimodal fusion.

### The "Skip" Problem
Traditional systems rely on broad genre tags or popularity. We solve this by creating a **mathematical "ear"** for sound, ensuring that recommendations are acoustically aligned with the seed song's timbre, texture, and energy.

---

## 🧠 Core Architecture: Hybrid Retrieval
We fuse two distinct modalities using a weighted distance ensemble:

### 1. Distance Metric
`Score = MSE(Metadata_Dist) + λ × (1 − Cosine_Similarity(Audio_Embeddings))`

*   **Metadata (8D):** High-level vibes (Danceability, Energy, etc.). MSE captures absolute intensity differences.
*   **Audio Embedding (56D):** Raw MFCC, Chroma, and Spectral features. Cosine Similarity captures the *direction* of the acoustic signature, overcoming the curse of dimensionality.

### 2. Preventing "Cosine Collapse"
Raw Librosa features vary wildly in scale (MFCC_0: -300, ZCR: 0.05). If unscaled, the largest dimensions dominate the vector angle, making all songs look identical. 
**Our Solution:** We enforce **Z-score Standardization (Zero Mean, Unit Variance)** across the entire 13k+ song dataset. This ensures every acoustic property has an equal voice, resulting in a balanced similarity space.

---

## ⚖️ λ-Tuning & Evaluation (Proven Merit)
We don't guess our parameters. We tune them via a **Same-Artist vs Random-Pair Discrimination Sweep**.

### Results (on 13,162 unique tracks)
| Variant | Pass Rate (Discrimination Accuracy) |
| :--- | :--- |
| Random Baseline | 50.0% |
| Metadata-only | 76.7% |
| Audio-only | 76.0% |
| **Hybrid (λ=0.1)** | **83.3%** ✅ |

*   **Pass Rate:** The probability that a song from the same artist is ranked closer than a completely random song.
*   **Chosen λ = 0.1:** Maximizes the engine's ability to recognize consistent production styles and vocal timbres.

---

## 🚀 Key Features & Interactive UI
*   **🎧 Seed Song Tab:** Discover similarity with premium visualizations.
*   **🔍 Analyze Any Song:** Zero-text inference. Download any YouTube URL, extract its 56D DNA, and find its matches in our 13k database.
*   **🎚️ Live λ Slider:** Side-by-side control. Shift from "Metadata-focused" (Balanced) to "Sound-focused" (Pure Audio) in real-time.
*   **📼 Global Music Player:** YouTube-style persistent player with a unified queue. Starts with the seed song and flows seamlessly into recommendations.
*   **📊 3D PCA Discovery:** Explore the 13,162-song vector space in interactive 3D, projected via our manual SVD implementation.

---

## 🗂️ Project Structure
```
src/
  pca.py               # [MANUAL] SVD-based PCA implementation
  clustering.py        # [MANUAL] Lloyd's algorithm & K-Means++ implementation
  recommender.py       # Hybrid distance logic (MSE + Cosine)
  tune_lambda.py       # λ tuning sweep & evaluation scripts
  audio_feature_extractor.py  # 56D Librosa feature engineering
  fetch_youtube_audio.py      # Automated nightly download pipeline
  build_master_dataset.py     # Data fusion & Z-score scaling

app/
  streamlit_app.py     # Premium UI (YouTube Music style)

data/
  dataset/             # master_music_data.parquet (Streamlit App DB)
  audio_files/         # Raw .m4a files from YouTube (git-ignored)
  embeddings/          # 58D Librosa extractions (git-ignored)
```

---

## 🛠️ Setup & Usage
1. **Install:** `pip install -r requirements.txt`
2. **Launch:** `python -m streamlit run app/streamlit_app.py`
3. **Reproduce Eval:** `python3 src/tune_lambda.py`

---

## 👥 Division of Labor
| Team Member | Primary Responsibility |
| :--- | :--- |
| **Aiden** | Data engineering — YouTube pipeline, Librosa feature extraction |
| **Kai** | UI/UX Design, JavaScript/CSS, Global Player integration |
| **Max** | Manual PCA (SVD) & Clustering implementation |
| **Yanfu** | Distance metric design, λ tuning methodology |
| **Sue** | Technical defense, Evaluation metrics & Sanity checks |
