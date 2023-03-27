from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec, FastText

nltk.download('stopwords')
stop_words = stopwords.words('english')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

df = pd.read_csv('./UrbanSound8K.csv')
class_column = df['class']
audio_names = class_column.tolist()
filename_column = df['slice_file_name']
filenames = filename_column.tolist()

processed_audio_names = [preprocess_text(name) for name in audio_names]

model = Word2Vec(sentences=processed_audio_names, min_count=1, vector_size=100, window=5, workers=4)

class_to_filename = dict(zip(audio_names, filenames))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results")
def results():
    search_term = request.args.get("search_term")
    if not search_term:
        return render_template("index.html")

    input_text = search_term.lower()
    input_words = input_text.split()

    similar_results = []
    for input_word in input_words:
        if input_word in model.wv.key_to_index:
            similar_words = model.wv.most_similar(positive=[input_word], topn=5)
            similar_files = [(class_to_filename[word], similarity) for word, similarity in similar_words]
            similar_results += similar_files

    audio_files = []
    for file, similarity in similar_results:
        audio_file = {"filename": file, "url": f"./audio_files/{file}"}
        if audio_file not in audio_files:
            audio_files.append(audio_file)

    return render_template("results.html", audio_files=audio_files, search_term=search_term)

if __name__ == '__main__':
    app.run(debug=True)
