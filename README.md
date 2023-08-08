# Text Classification for Service Prediction

This project demonstrates text classification using machine learning techniques to predict service keywords based on input sentences. It involves preprocessing the data, training a model, and utilizing it for prediction.

## Prerequisites

- Python 3.7 or above
- Required Python packages can be installed using `pip install -r requirements.txt`

## Project Structure

- `trained_model.pkl`: Serialized trained model for text classification.
- `vectorizer.pkl`: Serialized TF-IDF vectorizer used for feature extraction.
- `service.json`: JSON file containing service data with keywords and associated information.
- `preprocess_files.py`: Script to preprocess text files in the 'datas' directory by removing punctuation and converting content to lowercase.
- `train_text_classifier.py`: Script to train a text classification model using the preprocessed data.
- `predict_keyword_service.py`: Script to predict service keywords based on user input.

## Usage

1. Place text files to be preprocessed in the 'datas' directory.
2. Run `preprocess_files.py` to remove punctuation and convert content to lowercase in the text files.
3. Run `train_text_classifier.py` to train the text classification model using the preprocessed data.
4. Use `predict_keyword_service.py` to predict service keywords for input sentences.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your suggested changes.

