import nltk
import pickle
from nltk import tag
import numpy as np
import json
import random
import string
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nltk.download("punkt")
except:
    sys.exit(1)
# Load model dan variabel lain
model = load_model('model.h5')

file_data = open('data_chatbot.json', encoding='utf-8', errors="ignore").read()
intents = json.loads(file_data)
kata_tanya = pickle.load(open('kata_tanya.pkl','rb'))
kata_tag = pickle.load(open('kata_tag.pkl','rb'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def konverter_kalimat(kalimat, kata):
    kalimat_clean = [' '.join(stemmer.stem(w.lower()) for w in nltk.word_tokenize(kalimat) if w not in string.punctuation)]
    tfidf = TfidfVectorizer(vocabulary=kata, stop_words={'indonesian'})
    vector = tfidf.fit_transform(kalimat_clean)
    vector = vector.toarray()
    vector = vector.flatten()
    return vector

# print(konverter_kalimat('akun potongan lain-lain pada SPM kekurangan gaji', kata=kata_tanya))

def prediksi_jawaban(kalimat, model):
    prediksi = konverter_kalimat(kalimat, kata=kata_tanya)
    output = model.predict(np.array([prediksi]))[0]
    ERROR_THRESHOLD = 0.30
    vector_hasil = [[index,kata] for index, kata in enumerate(output) if kata > ERROR_THRESHOLD]
    vector_hasil.sort(key=lambda x: x[1], reverse=True) # mengurutkan hasil dari yang terbesar
    hasil_vector = []
    for hasil in vector_hasil:
        hasil_vector.append({'tag': kata_tag[hasil[0]], 'akurasi':str(hasil[1])})
    return hasil_vector

# print(prediksi_jawaban('bagaimana cara merekam saldo awal persediaan sakti', model))

def ambil_jawaban(tanya, intents_json):
    try:
        tag = tanya[0]['tag']
        vector_tag = intents_json['intents']
        for t in vector_tag:
            if t['tag'] == tag:
                jawab = t['responses']
                break
    except:
        jawab = 'Maaf coba gunakan kalimat lain'
    return jawab

def respon_chatbot(pesan):
    tanya = prediksi_jawaban(pesan, model)
    respon = ambil_jawaban(tanya, intents)
    return respon

print(prediksi_jawaban('cara melakukan penghapusan persediaan',model))
print(respon_chatbot('cara melakukan penghapusan persediaan'))

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return respon_chatbot(userText)


if __name__ == "__main__":
    app.run()
