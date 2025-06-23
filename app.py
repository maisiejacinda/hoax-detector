import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import re
import os
import gdown
import traceback

# ===================== Konfigurasi Streamlit =====================
st.set_page_config(page_title="Deteksi Berita Hoax", layout="wide")
st.title("📰 Deteksi Berita Hoax Indonesia")
st.markdown("Masukkan isi teks berita untuk dideteksi apakah valid atau hoax.")

# ===================== Clean Text =====================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# ===================== Model Arsitektur =====================
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)  # Penting: sama seperti saat training
        logits = self.fc(x)
        return logits

# ===================== Caching Bert Components =====================
@st.cache_resource(show_spinner=False)
def load_bert_components():
    with st.spinner("⏳ Memuat tokenizer & IndoBERT..."):
        tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
        bert_model = BertModel.from_pretrained("indobenchmark/indobert-base-p1")
    return tokenizer, bert_model

# ===================== Download & Cache Model Weights =====================
@st.cache_resource(show_spinner=False)
def download_and_load_custom_model_weights(file_id, destination):
    if not os.path.exists(destination):
        with st.spinner("⏳ Mengunduh model dari Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)
    return torch.load(destination, map_location="cpu")

# ===================== Load Semua Komponen Sekali Saja =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"🖥️ Perangkat aktif: **{device.type.upper()}**")

FILE_ID = "1mV18jc7zKTu_DFPENPYs7yGHrt2Fri_a"
MODEL_PATH = "model_hoax_final.pt"

tokenizer, bert_model = load_bert_components()
bert_model.to(device)

model = IndoBERT_CNN_LSTM(bert_model)
state_dict = download_and_load_custom_model_weights(FILE_ID, MODEL_PATH)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ===================== Input & Deteksi =====================
input_text = st.text_area("📋 Masukkan Teks Berita", height=200)

if st.button("🔍 Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    elif "http" in input_text:
        st.warning("Masukkan isi berita, bukan URL.")
    else:
        try:
            cleaned = clean_text(input_text)
            st.markdown("---")
            st.write("🧼 Teks Setelah Dibersihkan:")
            st.markdown(f"> *{cleaned}*")

            tokens = tokenizer(
                cleaned,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            )

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

            st.subheader("📊 Hasil Deteksi:")
            st.json({
                "Valid": round(probs[0][0].item(), 4),
                "Hoax": round(probs[0][1].item(), 4)
            })

            if confidence < 0.6:
                st.warning("⚠️ Model kurang yakin dengan prediksi ini. Harap waspada.")

            if pred == 1 and confidence >= 0.6:
                st.error(f"❌ Prediksi: HOAX – Confidence: {confidence:.2f}")
            else:
                st.success(f"✅ Prediksi: Valid – Confidence: {confidence:.2f}")

        except Exception as e:
            st.error("🚨 Terjadi error saat memproses input.")
            st.code(traceback.format_exc())
