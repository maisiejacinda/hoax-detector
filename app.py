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
st.markdown("Masukkan isi teks berita di bawah ini:")

input_text = st.text_area("📋 Teks Berita", height=200)

if st.button("🔍 Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    elif "http" in input_text:
        st.warning("Masukkan isi berita, bukan URL.")
    else:
        try:
            device = torch.device("cpu")
            download_model_from_drive("1p4wrI6A3i0GLKhACAYUDSu61LtFQB_kJ", "model_hoax.pt")

            bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
            bert_model = bert_model.to(device)

            model = IndoBERT_CNN_LSTM(bert_model)
            model.load_state_dict(torch.load("model_hoax.pt", map_location=device))
            model = model.to(device)
            model.eval()

            tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

            cleaned = clean_text(input_text)
            st.write("🧽 Teks setelah dibersihkan:", cleaned)

            tokens = tokenizer(cleaned, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            st.write("📏 Jumlah token:", input_ids.shape[1])

            with torch.no_grad():
                output = model(input_ids, attention_mask)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()
                confidence_valid = probs[0][0].item()
                confidence_hoax = probs[0][1].item()

                st.write(f"📊 Confidence Valid: {confidence_valid:.2f}")
                st.write(f"📊 Confidence Hoax: {confidence_hoax:.2f}")

                # === Override valid (berita protokol atau vaksin resmi)
                override_valid_keywords = ["masker", "protokol", "pemerintah", "kesehatan", "vaksinasi"]
                valid_triggered = any(word in cleaned for word in override_valid_keywords)

                # === Override hoax (berita palsu berbahaya)
                override_hoax_keywords = ["logam berat", "chip", "mikrochip", "mengontrol pikiran", "tanpa efek samping", "konspirasi", "sumber tak dikenal"]
                hoax_triggered = any(word in cleaned for word in override_hoax_keywords)

                # === Final logic
                if hoax_triggered and confidence < 0.7:
                    st.warning("⚠️ Model ragu, tapi teks ini mengandung klaim berbahaya.")
                    st.error(f"❌ Berita terindikasi Hoax – Confidence rendah: {confidence:.2f}")
                elif valid_triggered and pred == 1:
                    st.warning("⚠️ Deteksi otomatis menyebut 'Hoax', namun mengandung kata-kata kesehatan atau resmi.")
                    st.info(f"Prediksi awal: ❌ Hoax – Confidence: {confidence:.2f}")
                elif confidence < 0.55:
                    st.warning("⚠️ Model tidak yakin penuh. Hasil mendekati netral.")
                elif pred == 1 and confidence >= 0.70:
                    st.error(f"❌ Berita terindikasi Hoax – Confidence: {confidence:.2f}")
                else:
                    st.success(f"✅ Berita Valid – Confidence: {confidence:.2f}")

        except Exception as e:
            st.error("❌ Terjadi error saat deteksi.")
            st.code(traceback.format_exc())
