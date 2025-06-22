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
    gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

# === Fungsi Pembersih Teks ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", '', text)
    text = re.sub(r"\d+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

# === Arsitektur Model Custom ===
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

# === Inisialisasi Model dan Tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
download_model_from_drive("1z_dUz9Dcw4oR2LA7n9Lh55eTucMemNya", "model_hoax.pt")

bert_model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
bert_model = bert_model.to(device)

model = IndoBERT_CNN_LSTM(bert_model)
model.load_state_dict(torch.load("model_hoax.pt", map_location=device))
model = model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

# === Tampilan Streamlit ===
st.set_page_config(page_title="Deteksi Berita Hoax", layout="wide")
st.title("📰 Aplikasi Deteksi Berita Hoax Indonesia")

st.markdown("""
Masukkan teks berita di bawah ini untuk mendeteksi apakah termasuk hoax atau valid.  
Sistem akan menampilkan hasil deteksi **beserta tingkat keyakinan (confidence)**.  
Jika model kurang yakin, hasil default akan dianggap valid sebagai tindakan keamanan.
""")

input_text = st.text_area("📋 Teks Berita", height=200)

if st.button("🔍 Deteksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    elif "http" in input_text:
        st.warning("Masukkan isi teks berita, bukan URL/link.")
    else:
        try:
            cleaned = clean_text(input_text)
            st.write("🧽 Teks setelah dibersihkan:", cleaned)

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

            with torch.no_grad():
                output = model(input_ids, attention_mask)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

                # FIX: override jika confidence rendah
                if confidence < 0.6:
                    label = "✅ Berita Valid (Confidence rendah)"
                    st.warning(f"{label} – Confidence: {confidence:.2f}")
                else:
                    label = "✅ Berita Valid" if pred == 0 else "❌ Berita Hoax"
                    st.success(f"{label} – Confidence: {confidence:.2f}")

  except Exception as e:
    import traceback
    st.error("❌ Terjadi error saat mendeteksi.")
    st.code(traceback.format_exc())  # 🔍 Tampilkan log lengkap!
