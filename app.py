# === streamlit_hoax_detector.py ===
import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import re
import os
import gdown
import traceback
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# === Download model dari Google Drive ===
def download_model_from_drive(file_id, destination):
    if os.path.exists(destination):
        return
    gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

# === Bersihkan teks ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# === Arsitektur model CNN + LSTM ===
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        logits = self.fc(h_n.squeeze(0))
        return logits

# === Cache model & tokenizer ===
@st.cache_resource
def load_model_and_tokenizer():
    bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = IndoBERT_CNN_LSTM(bert_model)
    model.load_state_dict(torch.load("model_hoax.pt", map_location=torch.device("cpu")))
    model.eval()
    return model, tokenizer

# === UI Setup ===
st.set_page_config(page_title="Deteksi Berita Hoax", layout="wide")
st.title("📰 Aplikasi Deteksi Berita Hoax Indonesia")
st.markdown("Masukkan isi teks atau judul berita di bawah ini:")

input_text = st.text_area("📋 Teks atau Judul Berita", height=200)
hasil_list = []

if st.button("🔍 Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    elif "http" in input_text:
        st.warning("Masukkan isi berita, bukan URL.")
    else:
        try:
            download_model_from_drive("1p4wrI6A3i0GLKhACAYUDSu61LtFQB_kJ", "model_hoax.pt")
            model, tokenizer = load_model_and_tokenizer()
            device = torch.device("cpu")

            cleaned = clean_text(input_text)
            st.write("🧽 Teks setelah dibersihkan:", cleaned)

            if len(cleaned.split()) <= 12:
                st.warning("⚠️ Ini sepertinya hanya judul atau teks terlalu pendek, hasil mungkin kurang akurat.")

            # Keyword Check
            trusted_sources = ["cnn indonesia", "kompas", "detik", "tempo", "antaranews"]
            override_valid_keywords = [
                "masker", "protokol", "pemerintah", "kesehatan", "vaksinasi", "kementerian", "resmi"
            ]
            override_hoax_keywords = [
                "lebih ampuh dari vaksin", "menggantikan vaksin", "cahaya matahari menyembuhkan",
                "logam berat", "chip", "mikrochip", "mengontrol pikiran", "tanpa efek samping",
                "konspirasi", "sumber tak dikenal", "melacak lokasi", "booster untuk chip",
                "dilacak", "satelit", "bawang putih", "air es", "vaksin menyebabkan",
                "who melarang", "masker beracun", "zat beracun dalam masker",
                "uang tunai menolak vaksin", "hadiah bagi penolak vaksin"
            ]
            valid_triggered = any(word in cleaned for word in override_valid_keywords)
            hoax_triggered = any(word in cleaned for word in override_hoax_keywords)
            source_triggered = any(source in cleaned for source in trusted_sources)

            if valid_triggered and hoax_triggered:
                st.warning("⚖️ Ditemukan klaim mencurigakan, namun juga kata-kata resmi pemerintah.")
                st.info("🟡 Hasil tidak pasti – periksa informasi lebih lanjut.")
                hasil_list.append(["Teks lengkap", cleaned, "Tidak Pasti", 0.5])
                st.stop()
            elif source_triggered:
                st.info("📣 Ditemukan nama sumber terpercaya. Menganggap berita ini valid.")
                st.success("✅ Berita Valid – berdasarkan sumber terpercaya")
                hasil_list.append(["Teks lengkap", cleaned, "Valid (sumber terpercaya)", 1.0])
                st.stop()
            elif hoax_triggered:
                st.warning("⚠️ Klaim yang sering dikaitkan dengan hoaks terdeteksi.")
                st.error("❌ Berita terindikasi Hoax – berdasarkan kata kunci mencurigakan")
                hasil_list.append(["Teks lengkap", cleaned, "Hoax (keyword)", 1.0])
                st.stop()

            # === Deteksi langsung ===
            tokens = tokenizer(cleaned, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            with torch.no_grad():
                output = model(input_ids, attention_mask)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence_valid = probs[0][0].item()
                confidence_hoax = probs[0][1].item()

            if probs[0][pred].item() < 0.55:
                st.warning("⚠️ Confidence rendah, model tidak yakin penuh.")

            label = "Valid" if pred == 0 else "Hoax"
            st.metric(label="📌 Hasil Prediksi", value=label, delta=f"{probs[0][pred].item():.2%}")
            st.write(f"📊 Confidence Valid: {confidence_valid:.2f}")
            st.write(f"📊 Confidence Hoax: {confidence_hoax:.2f}")

            hasil_list.append(["Teks lengkap", cleaned, label, probs[0][pred].item()])

        except Exception as e:
            st.error("❌ Terjadi error saat deteksi.")
            st.code(traceback.format_exc())

# === Export ke CSV ===
if hasil_list:
    df = pd.DataFrame(hasil_list, columns=["Bagian", "Teks", "Label", "Confidence"])
    st.subheader("📁 Hasil Deteksi")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil sebagai CSV", csv, "hasil_deteksi_hoaks.csv", "text/csv")
