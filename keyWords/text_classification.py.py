from warnings import simplefilter
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.svm import LinearSVC
import pymorphy2
import joblib
import os

simplefilter(action='ignore', category=FutureWarning)

# Initialize morphological analyzer
morph = pymorphy2.MorphAnalyzer()

directory = "datas"
files = defaultdict(list)
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_key = filename.split(".")[0]
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            files[file_key].extend(line.strip() for line in f.readlines())

# Create a dictionary 'files' to store sentences for each file based on file_key

training_labels = [keyword for keyword, file_sentences in files.items()
                   for _ in file_sentences]

sentences = [sentence for file_sentences in files.values()
             for sentence in file_sentences]


# Preprocess data
def preprocess_data(input_sentences):
    processed = []
    for sentence in input_sentences:
        normalized_tokens = [morph.parse(token)[0].normal_form
                             for token in sentence.lower().split()]
        processed.append(normalized_tokens)
    return processed

# Function to preprocess input sentences by tokenizing and normalizing them


# Compute TF-IDF
def compute_tfidf(processed):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_result = tfidf_vectorizer.fit_transform(
        [' '.join(tokens) for tokens in processed])
    return tfidf_vectorizer, tfidf_matrix_result


# Function to compute TF-IDF representation of the preprocessed sentences

processed_sentences = preprocess_data(sentences)
vectorizer, tfidf_matrix = compute_tfidf(processed_sentences)

# Preprocess sentences using the defined function and compute TF-IDF matrix

# Train the model
model = LinearSVC()
model.fit(tfidf_matrix, training_labels)

# Train a LinearSVC model using the TF-IDF matrix and training labels

# Save the trained model and vectorizer
joblib.dump(model, 'trained_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Save the trained model and vectorizer as pickle files
