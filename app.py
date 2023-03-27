# import machine learning libraries
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec, FastText
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

# link to dataset https://www.kaggle.com/datasets/chrisfilo/urbansound8k
df = pd.read_csv('../audio_files/UrbanSound8k.csv')
class_column = df['class']
audio_names = class_column.tolist()
class_to_filename = dict(zip(df['class'], df['slice_file_name']))
#print(audio_names[:5])

extended_audio_names = []

for name in audio_names:
    extended_audio_names.append(name.split('_'))

processed_audio_names = [preprocess_text(name) for name in audio_names] + extended_audio_names
#print(processed_audio_names[:5])
#print('car' in [word for words in processed_audio_names for word in words])
#print('car_horn' in [word for words in processed_audio_names for word in words])
# Train a Word2Vec model
model = Word2Vec(sentences=processed_audio_names, min_count=1, vector_size=100, window=5, workers=4)

# display model vocabulary
words = list(model.wv.key_to_index)

# display model vocabulary size
#print(words)
# print(len(set(words[:5])))
# print('********')



def get_mean_vector(words):
    vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None

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
        # Convert the rows to a list of tuples (file_name, fold)
        similar_files = list(similar_rows.itertuples(index=False, name=None))
        return similar_files[:topn]
    else:
        return []

# def jaccard_similarity(a, b):
#     a_set = set(a)
#     b_set = set(b)
#     intersection = a_set.intersection(b_set)
#     union = a_set.union(b_set)
#     return len(intersection) / len(union)

# def get_similar_words(input_text, topn=3):
#     input_words = preprocess_text(input_text)
#     scores = [(row.slice_file_name, row['class'], row.fold, jaccard_similarity(input_words, row['class'])) for index, row in df.iterrows()]
#     sorted_scores = sorted(scores, key=lambda x: x[3], reverse=True)[:topn]
#     return [(file_name, fold) for file_name, class_name, fold, score in sorted_scores]

# Returns 3 audio files and what fold they are in 

input_text = 'engine idling'
similar_results = get_similar_words(input_text, 3)
print(similar_results if similar_results else f"'{input_text}' not found in the dataset.")