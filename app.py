import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import re
import os
import gdown

# === Fungsi Download Model dari Google Drive ===
def download_model_from_drive(file_id, destination):
    if os.path.exists(destination):
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# === Fungsi Pembersih Teks ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# === Definisi Arsitektur Model (bert dikirim dari luar) ===
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

# === Load Model dan Tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download model jika belum ada
download_model_from_drive("1z_dUz9Dcw4oR2LA7n9Lh55eTucMemNya", "model_hoax.pt")

# Load pretrained IndoBERT dulu
bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
bert_model = bert_model.to(device)

# Kirim ke arsitektur custom
model = IndoBERT_CNN_LSTM(bert_model)
model.load_state_dict(torch.load("model_hoax.pt", map_location=device))
model = model.to(device)
model.eval()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

# === Tampilan Streamlit ===
st.set_page_config(page_title="Deteksi Berita Hoax", layout="wide")
st.title("📰 Aplikasi Deteksi Berita Hoax Indonesia")
st.markdown("Masukkan teks berita di bawah ini:")

input_text = st.text_area("📋 Teks Berita")

if st.button("🔍 Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        cleaned = clean_text(input_text)
        tokens = tokenizer(cleaned, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            pred = torch.argmax(output, dim=1).item()

        st.success("✅ Berita Valid" if pred == 0 else "❌ Berita Hoax")
