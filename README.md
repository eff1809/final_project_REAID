# 📚 AI Study Buddy

## 👤 Identitas Project
- **Nama**          : Efraim Imanuel Parasak
- **Final Project** : AI Study Buddy
- **Program**       : AI Python Bootcamp

---

## Deskripsi Singkat

**AI Study Buddy** adalah aplikasi berbasis **Streamlit dan HuggingFace Transformers**
yang berfungsi sebagai **pendamping belajar berbasis AI**.

Aplikasi ini membantu pengguna untuk:
- 📄 Merangkum materi belajar secara otomatis
- 💬 Menjawab pertanyaan berdasarkan konteks materi
- 🧪 Menghasilkan kuis dari materi belajar

Project ini dikembangkan sebagai **Final Project AI Python Bootcamp** dengan fokus
pada penerapan **Natural Language Processing (NLP)** dan **Large Language Models (LLM)**.

---

## Live App
**Live Demo:**
https://finalprojectreaid-aipythonbootcampskillacademy.streamlit.app/

---

## GitHub Repository
**Source Code:**
https://github.com/eff1809/final_project_REAID

---

## Fitur Utama

### 1️⃣ Ringkas Materi
Merangkum teks panjang seperti artikel atau catatan kuliah menjadi ringkasan singkat
dan mudah dipahami menggunakan model **facebook/bart-large-cnn**.

---

### 2️⃣ Tanya Jawab dengan AI
Menjawab pertanyaan berdasarkan materi yang dimasukkan pengguna dengan pendekatan:
- **Extractive Question Answering** (RoBERTa SQuAD2)
- **Generative Fallback Answering** menggunakan **FLAN-T5** jika jawaban tidak ditemukan secara eksplisit

---

### 3️⃣ Generate Kuis Otomatis
Menghasilkan soal kuis dari materi belajar secara otomatis untuk membantu proses evaluasi belajar,
menggunakan model **google/flan-t5-base**.

---

## Teknologi yang Digunakan

- **Python** 3.10
- **Streamlit**
- **HuggingFace Transformers**
- **PyTorch**
- **Model AI**:
  - facebook/bart-large-cnn (Summarization)
  - deepset/roberta-base-squad2 (Question Answering)
  - google/flan-t5-base (Quiz Generation)

---

## Cara Menjalankan Secara Lokal

```bash
git clone https://github.com/eff1809/final_project_REAID.git
cd final_project_REAID
pip install -r requirements.txt
streamlit run app.py
