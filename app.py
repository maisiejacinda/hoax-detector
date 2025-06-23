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

            trusted_sources = ["cnn indonesia", "kompas", "detik", "tempo", "antaranews"]
            override_valid_keywords = [
                "presiden", "jokowi", "kementerian", "resmi", "pemerintah", "bmkg", "gratis", "pengumuman",
                "kompas", "cnn indonesia", "detik", "tempo", "sumber resmi", "puskesmas", "booster",
                "rakyat", "wilayah", "diluncurkan", "peresmian", "meresmikan"
            ]
            override_hoax_keywords = [
                "logam berat", "chip", "mikrochip", "mengontrol pikiran", "tanpa efek samping",
                "konspirasi", "sumber tak dikenal", "melacak lokasi", "booster untuk chip",
                "dilacak", "satelit", "bawang putih", "sembuh dalam semalam", "air es", "vaksin menyebabkan",
                "bill gates", "dikendalikan", "matahari lebih ampuh", "air kelapa menyembuhkan"
            ]

            if any(source in cleaned for source in trusted_sources):
                st.info("📣 Ditemukan nama sumber terpercaya. Menganggap berita ini valid.")
                st.success("✅ Berita Valid – berdasarkan sumber terpercaya")
                hasil_list.append(["Teks lengkap", cleaned, "Valid (sumber terpercaya)", 1.0])
                st.stop()

            if any(word in cleaned for word in override_valid_keywords):
                st.success("✅ Berita Valid – berdasarkan kata kunci resmi")
                hasil_list.append(["Teks lengkap", cleaned, "Valid (keyword resmi)", 1.0])
                st.stop()

            if any(word in cleaned for word in override_hoax_keywords):
                st.warning("⚠️ Klaim yang sering dikaitkan dengan hoaks terdeteksi.")
                st.error("❌ Berita terindikasi Hoax – berdasarkan kata kunci mencurigakan")
                hasil_list.append(["Teks lengkap", cleaned, "Hoax (keyword)", 1.0])
                st.stop()

            tokens = tokenizer(cleaned, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            with torch.no_grad():
                output = model(input_ids, attention_mask)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence_valid = probs[0][0].item()
                confidence_hoax = probs[0][1].item()
                confidence = probs[0][pred].item()

            gap = abs(confidence_valid - confidence_hoax)
            label = "Valid" if pred == 0 else "Hoax"

            if gap < 0.07:
                st.warning("⚖️ Model tidak yakin – hasil mendekati netral.")
                label = "Tidak Pasti"
            elif confidence < 0.6:
                st.warning("⚠️ Confidence rendah, model tidak yakin penuh.")

            st.metric(label="📌 Hasil Prediksi", value=f"{label}", delta=f"{confidence:.2%}")
            st.write(f"📊 Confidence Valid: {confidence_valid:.2f}")
            st.write(f"📊 Confidence Hoax: {confidence_hoax:.2f}")

            hasil_list.append(["Teks lengkap", cleaned, label, confidence])

        except Exception as e:
            st.error("❌ Terjadi error saat deteksi.")
            st.code(traceback.format_exc())

if hasil_list:
    df = pd.DataFrame(hasil_list, columns=["Bagian", "Teks", "Label", "Confidence"])
    st.subheader("📁 Hasil Deteksi")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil sebagai CSV", csv, "hasil_deteksi_hoaks.csv", "text/csv")
