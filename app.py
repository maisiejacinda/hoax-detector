import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import re
import os
import gdown
import traceback

# =============== CACHING KOMONEN AGAR CEPAT ===============
@st.cache_resource(show_spinner=False)
def download_and_load_model_weights(file_id, destination, device):
    if not os.path.exists(destination):
        with st.spinner("⏳ Mengunduh model dari Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)
    model = IndoBERT_CNN_LSTM(load_bert_components(device))
    model.load_state_dict(torch.load(destination, map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_bert_components(device):
    with st.spinner("⏳ Memuat tokenizer dan IndoBERT..."):
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p1').to(device)
    return tokenizer, bert_model

# =============== BERSIHKAN TEKS ===============
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# =============== DEFINISI MODEL SESUAI TRAINING ===============
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self, components):
        super().__init__()
        tokenizer, bert = components
        self.tokenizer = tokenizer
        self.bert = bert
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)  # ⛔ Jangan ubah, ini sesuai training model
        logits = self.fc(x)
        return logits

# =============== STREAMLIT UI ===============
st.set_page_config(page_title="Deteksi Berita Hoax", layout="wide")
st.title("📰 Deteksi Berita Hoax Indonesia")
st.markdown("Masukkan isi teks berita untuk dideteksi apakah valid atau hoax.")

input_text = st.text_area("📋 Masukkan Teks Berita", height=200)

if st.button("🔍 Deteks
