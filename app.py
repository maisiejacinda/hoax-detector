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

# === Fungsi Pembersih Teks ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# === Arsitektur Model CNN + LSTM + IndoBERT ===
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state                # [batch_size, seq_len, hidden_size]
        x = x.permute(0, 2, 1)                           # [batch_size, hidden_size, seq_len]
        x = self.conv1(x)                                # [batch_size, 128, seq_len]
        x = x.permute(0, 2, 1)                           # [batch_size, seq_len, 128]
        _, (h_n, _) = self.lstm(x)                       # h_n: [1, batch_size, 64]
        logits = self.fc(h_n.squeeze(0))                 # [batch_size, 2]
        return logits

# === Streamlit UI ===
st.set_page_config(page_title="Deteksi Berita Hoax", layout="wide")
st.title("📰 Deteksi Berita Hoax Indonesia")
st.markdown("Masukkan isi teks berita (bukan link) untuk dicek apakah hoax atau tidak.")

input_text = st.text_area("📋 Teks Berita", height=200)

if st.button("🔍 Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    elif "http" in input_text:
        st.warning("Masukkan teks berita, bukan URL/link.")
    else:
        try:
            st.info("⏳ Memuat model dan tokenizer...")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # === Download & Load Model ===
            model_path = "model_hoax.pt"
            download_model_from_drive("1PKeq5RktCLpDnueDzQgr-khNYqTeTpmR", model_path)

            # Load tokenizer & IndoBERT
            tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
            bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
            bert_model.to(device)

            model = IndoBERT_CNN_LSTM(bert_model)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # === Preprocessing & Tokenisasi ===
            cleaned_text = clean_text(input_text)
            st.write("🧼 Teks setelah dibersihkan:", cleaned_text)

            tokens = tokenizer(
                cleaned_text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            )

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            # === Prediksi ===
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

            st.write("📊 Probabilitas:", { "Valid": float(probs[0][0]), "Hoax": float(probs[0][1]) })

            # === Output Prediksi ===
            if pred == 1 and confidence >= 0.60:
                st.error(f"❌ Prediksi: Berita Hoax – Confidence: {confidence:.2f}")
            else:
                st.success(f"✅ Prediksi: Berita Valid – Confidence: {confidence:.2f}")

        except Exception as e:
            st.error("🚨 Terjadi error saat mendeteksi.")
            st.code(traceback.format_exc())
