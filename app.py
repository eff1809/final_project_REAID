# =========================================
# AI Study Buddy
# Berisi:
# 1. UI (Streamlit)
# 2. Logic Aplikasi
# 3. Class & Function (OOP)
# =========================================

import streamlit as st
from transformers import pipeline
from dataclasses import dataclass


# ==============================
# CONFIGURATION (OOP)
# ==============================

@dataclass
class ModelConfig:
    summarization_model: str = "facebook/bart-large-cnn"
    qa_model: str = "deepset/roberta-base-squad2"
    quiz_model: str = "google/flan-t5-base"


# ==============================
# AI CORE LOGIC (OOP)
# ==============================

class StudyBuddyAI:
    def __init__(self, config: ModelConfig):
        self.summarizer = pipeline(
            "summarization", model=config.summarization_model
        )
        self.qa = pipeline(
            "question-answering", model=config.qa_model
        )
        self.generator = pipeline(
            "text2text-generation", model=config.quiz_model
        )

    def summarize(self, text: str) -> str:
        result = self.summarizer(
            text,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        return result[0]["summary_text"]

    def ask(self, context: str, question: str) -> str:
        result = self.qa(
            question=question,
            context=context
        )
        return result["answer"]

    def generate_quiz(self, text: str) -> str:
        prompt = (
            "Create 5 short quiz questions based on the following study material:\n"
            f"{text}"
        )
        result = self.generator(
            prompt,
            max_length=256,
            do_sample=False
        )
        return result[0]["generated_text"]


# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Study Buddy")
st.write(
    "AI pendamping belajar untuk **merangkum materi**, "
    "**tanya jawab**, dan **membuat kuis otomatis**."
)

# Initialize AI
config = ModelConfig()
ai = StudyBuddyAI(config)

st.header("📝 Input Materi Belajar")
material = st.text_area(
    "Masukkan materi (catatan kuliah, artikel, dll)",
    height=250
)

# Optional safety check
if material and len(material.split()) > 800:
    st.warning(
        "Materi cukup panjang. Untuk hasil terbaik, "
        "pertimbangkan merangkum sebagian terlebih dahulu."
    )

st.divider()

col1, col2, col3 = st.columns(3)

# --------- SUMMARIZATION ---------
with col1:
    if st.button("📄 Ringkas Materi"):
        if material.strip():
            with st.spinner("Merangkum materi..."):
                summary = ai.summarize(material)
                st.success(summary)
        else:
            st.warning("Materi tidak boleh kosong.")

# --------- QUESTION ANSWERING ---------
with col2:
    question = st.text_input(
        "💬 Masukkan pertanyaan tentang materi"
    )

    if st.button("Tanya AI"):
        if material.strip() and question.strip():
            with st.spinner("Mencari jawaban..."):
                answer = ai.ask(material, question)
                st.info(answer)
        else:
            st.warning(
                "Materi dan pertanyaan tidak boleh kosong."
            )

# --------- QUIZ GENERATION ---------
with col3:
    if st.button("🧪 Buat Kuis"):
        if material.strip():
            with st.spinner("Membuat kuis..."):
                quiz = ai.generate_quiz(material)
                st.write(quiz)
        else:
            st.warning("Materi tidak boleh kosong.")

st.divider()
st.caption(
    "Final Project AI | Streamlit + HuggingFace Transformers"
)
