import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import re
import os
import gdown
import traceback

# === Fungsi Download Model dari Google Drive ===
def download_model_from_drive(file_id, destination):
    if os.path.exists(destination):
        return
    gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

# === Pembersih Teks ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# === Arsitektur Model CNN + LSTM ===
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

# === Tampilan UI Streamlit ===
st.set_page_config(page_title="Deteksi Berita Hoax", layout="wide")
st.title("📰 Aplikasi Deteksi Berita Hoax Indonesia")
st.markdown("Masukkan isi teks berita di bawah ini (bukan URL/link):")

input_text = st.text_area("📋 Teks Berita", height=200)

if st.button("🔍 Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    elif "http" in input_text:
        st.warning("Masukkan isi berita, bukan URL.")
    else:
        try:
            st.info("⏳ Memuat model dan tokenizer...")

            device = torch.device("cpu")

            # Download model dari Google Drive
            download_model_from_drive("1p4wrI6A3i0GLKhACAYUDSu61LtFQB_kJ", "model_hoax.pt")

            # Load pretrained IndoBERT
            bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
            bert_model = bert_model.to(device)

            # Load arsitektur + bobot model
            model = IndoBERT_CNN_LSTM(bert_model)
            model.load_state_dict(torch.load("model_hoax.pt", map_location=device))
            model = model.to(device)
            model.eval()

            # Tokenizer
            tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

            # Bersihkan teks
            cleaned = clean_text(input_text)
            st.write("🧽 Teks setelah dibersihkan:", cleaned)

            # Tokenisasi
            tokens = tokenizer(
                cleaned,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            )

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            st.write("📏 Jumlah token:", input_ids.shape[1])

            # Prediksi
            with torch.no_grad():
                output = model(input_ids, attention_mask)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

                # Logika Aman: hanya prediksi HOAX jika confidence cukup tinggi
                if pred == 1 and confidence >= 0.60:
                    st.error(f"❌ Berita Hoax – Confidence: {confidence:.2f}")
                else:
                    st.success(f"✅ Berita Valid – Confidence: {confidence:.2f}")

        except Exception as e:
            st.error("❌ Terjadi error saat memproses.")
            st.code(traceback.format_exc())
