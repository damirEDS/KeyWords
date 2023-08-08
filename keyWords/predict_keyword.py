import joblib
import pymorphy2
import re
import json

# Load the trained model and vectorizer
model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize morphological analyzer
morph = pymorphy2.MorphAnalyzer()

# Load service data
with open('service.json', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Cache for morphological analysis results
morph_cache = {}


def remove_punctuation(sentence):
    # Function to remove punctuation from a sentence using regex
    processed_sentence = re.sub(r'[^\w\s]', '', sentence)
    return processed_sentence


# Preprocess data
def preprocess_data(input_sentences):
    processed = []
    for sentence in input_sentences:
        normalized_tokens = []
        for token in sentence.lower().split():
            if token not in morph_cache:
                morph_cache[token] = morph.parse(token)[0].normal_form
            normalized_tokens.append(morph_cache[token])
        processed.append(normalized_tokens)
    return processed


def predict_keyword(input_sentence):
    # Function to predict a keyword for an input sentence
    processed_input_sentence = preprocess_data([input_sentence])[0]
    tfidf_input = vectorizer.transform([' '.join(processed_input_sentence)])
    predicted_keyword = model.predict(tfidf_input)[0]
    return predicted_keyword


inputSentence = "проблема co принтером"
predictedKeyword = predict_keyword(inputSentence)
# If the key is found, return the value of the "service" property
service_value = data[predictedKeyword]["service"]
print(service_value)
print("Predicted keyword:", predictedKeyword)
