from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')

# function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    # tokenize text
    text = nltk.word_tokenize(text)
    return text

df = pd.read_csv('../audio_files/UrbanSound8k.csv')
class_column = df['class']
audio_names = class_column.tolist()
class_to_filename = dict(zip(df['class'], df['slice_file_name']))
#print(audio_names[:5])

extended_audio_names = []

for name in audio_names:
    extended_audio_names.append(name.split('_'))

processed_audio_names = [preprocess_text(name) for name in audio_names] + extended_audio_names

# Train a Word2Vec model
model = Word2Vec(sentences=processed_audio_names, min_count=1, vector_size=100, window=5, workers=4)

# display model vocabulary
words = list(model.wv.key_to_index)



def get_similar_words(input_text, topn=3):
    input_words = preprocess_text(input_text)
    
    class_scores = {class_name: 0 for class_name in set(df['class'])}
    
    for word in input_words:
        if word in model.wv.key_to_index:
            for class_name in class_scores.keys():
                similarity = model.wv.similarity(word, class_name)
                class_scores[class_name] = max(class_scores[class_name], similarity)
    
    sorted_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    similar_classes = [class_name for class_name, _ in sorted_scores]
    
    if similar_classes:
        # Select only rows with the similar classes
        similar_rows = df[df['class'].isin(similar_classes)][['slice_file_name', 'fold']]
        # Replace any remaining backslashes with forward slashes
        similar_rows['slice_file_name'] = similar_rows['slice_file_name'].str.replace('\\', '/')
        # Generate full file path of the audio files
        audio_files = []
        for file_name, fold in similar_rows.itertuples(index=False):
            audio_file_path = os.path.join('audio_files', 'static', f'fold{fold}', file_name)
            audio_files.append((file_name, audio_file_path))
        return audio_files[:topn]
    else:
        return []



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        similar_files = get_similar_words(input_text, topn=3)
        return render_template('index.html', results=similar_files)
    else:
        return render_template('index.html')
    
# Define route for serving audio files
@app.route('/audio_files/<path:file_path>')
def get_audio(file_path):
    formatted_file_path = file_path.replace('\\', '/')
    return send_file(formatted_file_path, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True)
