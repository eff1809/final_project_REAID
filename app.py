# =========================================
# AI Study Buddy (FINAL FIX VERSION)
# Streamlit + HuggingFace + OOP
# =========================================

import streamlit as st
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)
from dataclasses import dataclass


# ==============================
# CONFIGURATION (OOP)
# ==============================

@dataclass
class ModelConfig:
    summarization_model: str = "facebook/bart-large-cnn"
    qa_model: str = "deepset/roberta-base-squad2"
    quiz_model: str = "google/flan-t5-base"
    device: int = -1  # CPU ONLY (AMAN)


# ==============================
# MODEL LOADER (CACHED)
# ==============================

@st.cache_resource
def load_models(config: ModelConfig):
    # --- SUMMARIZATION ---
    sum_tokenizer = AutoTokenizer.from_pretrained(
        config.summarization_model
    )
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.summarization_model,
        low_cpu_mem_usage=False
    )

    summarizer = pipeline(
        "summarization",
        model=sum_model,
        tokenizer=sum_tokenizer,
        device=config.device
    )

    # --- QUESTION ANSWERING ---
    qa_tokenizer = AutoTokenizer.from_pretrained(
        config.qa_model
    )
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        config.qa_model,
        low_cpu_mem_usage=False
    )

    qa_pipeline = pipeline(
        "question-answering",
        model=qa_model,
        tokenizer=qa_tokenizer,
        device=config.device
    )

    # --- QUIZ GENERATION ---
    quiz_tokenizer = AutoTokenizer.from_pretrained(
        config.quiz_model
    )
    quiz_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.quiz_model,
        low_cpu_mem_usage=False
    )

    quiz_generator = pipeline(
        "text2text-generation",
        model=quiz_model,
        tokenizer=quiz_tokenizer,
        device=config.device
    )

    return summarizer, qa_pipeline, quiz_generator


# ==============================
# AI CORE LOGIC (OOP)
# ==============================

class StudyBuddyAI:
    def __init__(self, config: ModelConfig):
        (
            self.summarizer,
            self.qa,
            self.generator
        ) = load_models(config)

        # tokenizer khusus summarization
        self.sum_tokenizer = AutoTokenizer.from_pretrained(
            config.summarization_model
        )
    
    #fungsi chungking
    def _chunk_text(self, text: str, max_tokens: int = 900):
        """
        Memecah teks panjang menjadi beberapa chunk token-aman
        """
        tokens = self.sum_tokenizer(
            text,
            return_tensors="pt",
            truncation=False
        )["input_ids"][0]

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.sum_tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )
            chunks.append(chunk_text)

        return chunks
    def summarize(self, text: str) -> str:
        chunks = self._chunk_text(text)

        summaries = []

        for chunk in chunks:
            inputs = self.sum_tokenizer(
                chunk,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            )

            summary_ids = self.summarizer.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=50,
                do_sample=False
            )

            summary = self.sum_tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

            summaries.append(summary)

        # Gabungkan semua ringkasan
        final_summary = " ".join(summaries)

        if len(summaries) > 1:
            final_summary = self.summarizer(
                final_summary,
                max_length=180,
                min_length=80,
                do_sample=False
            )[0]["summary_text"]

        return final_summary
    
    def ask(self, context: str, question: str) -> str:
        """
        Smart QA Logic:
        1. Coba cari jawaban persis di teks (Extractive).
        2. Jika score rendah, gunakan Generative AI (LLM) dengan instruksi yang lebih tegas.
        """
        
        # --- LANGKAH 1: EXTRACTIVE QA ---
        # Kita pecah context jadi chunk kecil untuk RoBERTa
        chunks = self._chunk_text(context, max_tokens=350)
        best_answer = ""
        best_score = 0.0

        for chunk in chunks:
            try:
                result = self.qa(
                    question=question,
                    context=chunk
                )
                if result["score"] > best_score:
                    best_score = result["score"]
                    best_answer = result["answer"]
            except:
                continue

        # Ambang batas (Threshold):
        # Jika score > 0.5, kita cukup yakin jawabannya ada di teks.
        # Jika di bawah itu, kemungkinan jawabannya butuh pengetahuan umum/generatif.
        if best_score > 0.5:
            return f"Berdasarkan materi: {best_answer}"

        # --- LANGKAH 2: GENERATIVE QA (FALLBACK) ---
        # Kita gunakan FLAN-T5 jika jawaban tidak ketemu di teks secara eksplisit
        
        # Potong context agar tidak error di T5 (max 512 tokens input)
        # Kita ambil chunk pertama atau gabungan awal saja sebagai referensi
        truncated_context = context[:2000] # Ambil 2000 karakter pertama kira-kira
        
        # PROMPT ENGINEERING YANG DIPERBAIKI
        # Kita pisahkan instruksi dengan jelas agar model tidak mengulang teks
        prompt = (
            f"Instruksi: Jawab pertanyaan berikut dengan singkat dan jelas dalam Bahasa Indonesia. "
            f"Gunakan informasi dari Teks Referensi jika relevan. "
            f"Jika jawaban tidak ada di teks, gunakan pengetahuan umum.\n\n"
            f"Teks Referensi:\n{truncated_context}\n\n"
            f"Pertanyaan:\n{question}\n\n"
            f"Jawaban:"
        )

        try:
            gen_result = self.generator(
                prompt,
                max_length=200,     # Beri ruang untuk jawaban panjang
                do_sample=True,     # Sedikit variasi agar lebih natural
                temperature=0.3,    # Rendah agar tetap faktual
                repetition_penalty=1.2 # PENTING: Mencegah pengulangan teks input
            )
            return gen_result[0]["generated_text"]
        except Exception as e:
            return "Maaf, saya tidak dapat menemukan jawaban yang relevan."

    def generate_quiz(self, text: str) -> str:
        # Ambil ringkasan/potongan teks agar tidak terlalu panjang buat prompt
        short_text = text[:1500] 
        
        prompt = (
            "Buatkan 3 soal kuis pilihan ganda singkat berdasarkan teks berikut dalam Bahasa Indonesia:\n\n"
            f"{short_text}"
        )
        
        try:
            result = self.generator(
                prompt,
                max_length=300,
                do_sample=True,
                temperature=0.5
            )
            return result[0]["generated_text"]
        except:
            return "Gagal membuat kuis."


    # def generate_quiz(self, text: str) -> str:
    #     prompt = (
    #         "Create 5 short quiz questions based on this material:\n"
    #         f"{text}"
    #     )
    #     result = self.generator(
    #         prompt,
    #         max_length=256,
    #         do_sample=False
    #     )
    #     return result[0]["generated_text"]
    
    



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

st.info("📌 Model berjalan di CPU untuk stabilitas maksimum.")

config = ModelConfig()
ai = StudyBuddyAI(config)

st.header("📝 Input Materi Belajar")
material = st.text_area(
    "Masukkan materi (catatan kuliah, artikel, dll)",
    height=250
)

# if len(material.split()) > 800:
#     st.warning("Materi terlalu panjang, akan dipotong otomatis.")
    
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
    question = st.text_input("💬 Masukkan pertanyaan")

    if st.button("Tanya AI"):
        if material.strip() and question.strip():
            with st.spinner("Mencari jawaban..."):
                answer = ai.ask(material, question)
                st.info(answer)
        else:
            st.warning("Materi dan pertanyaan tidak boleh kosong.")

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
st.caption("Final Project AI | Streamlit + HuggingFace Transformers")
