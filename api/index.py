import pickle
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

app = Flask(__name__)

# Load model dan tokenizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 150  # sama seperti saat training

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ulasan = request.form.get("Ulasan")
    ulasan_clean = preprocess_text(ulasan)

    seq = tokenizer.texts_to_sequences([ulasan_clean])
    if not seq or not seq[0]:
        return render_template(
            "index.html",
            prediction_text="Ulasan kamu terlalu singkat atau kata-katanya tidak dikenali oleh model.",
        )

    pad = pad_sequences(seq, padding="post", maxlen=maxlen)

    pred = model.predict(pad)
    try:
        # Untuk Sklearn: prediksi langsung angka kelas
        label = int(pred[0])
    except:
        # Untuk Keras: prediksi probabilitas, ambil argmax
        import numpy as np
        label = int(np.argmax(pred, axis=1)[0])

    output_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    hasil = output_map.get(label, "Tidak diketahui")

    return render_template(
        "index.html", prediction_text=f"Hasil prediksi adalah: {hasil}"
    )

if __name__ == "__main__":
    app.run()
