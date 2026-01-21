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
    # def __init__(self, config: ModelConfig):
    #     (
    #         self.summarizer,
    #         self.qa,
    #         self.generator
    #     ) = load_models(config)

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
    # def summarize(self, text: str) -> str:
    #     result = self.summarizer(
    #         text,
    #         max_length=150,
    #         min_length=50,
    #         do_sample=False
    #     )
    #     return result[0]["summary_text"]
    
    # def summarize(self, text: str) -> str:
    # # POTONG INPUT AGAR TIDAK MELEBIHI 1024 TOKEN
    #     inputs = self.sum_tokenizer(
    #         text,
    #         max_length=1024,
    #         truncation=True,
    #         return_tensors="pt"
    #     )

    #     summary_ids = self.summarizer.model.generate(
    #         inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         max_length=150,
    #         min_length=50,
    #         do_sample=False
    #     )

    #     summary = self.sum_tokenizer.decode(
    #         summary_ids[0],
    #         skip_special_tokens=True
    #     )

    #     return summary
    
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
        result = self.qa(
            question=question,
            context=context
        )
        return result["answer"]

    def generate_quiz(self, text: str) -> str:
        prompt = (
            "Create 5 short quiz questions based on this material:\n"
            f"{text}"
        )
        result = self.generator(
            prompt,
            max_length=256,
            do_sample=False
        )
        return result[0]["generated_text"]
    
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
