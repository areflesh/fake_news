from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump, load
import os
import torch
import numpy as np
import requests
import zipfile
import logging
import pickle

logging.basicConfig(level=logging.INFO)

def load_glove_vectors(glove_file):
    """Load the GloVe vectors from a file."""
    download_glove(glove_file)
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings
    
def glove_features_generation(texts, embeddings, embedding_dim):
    """Create features from the texts using GloVe embeddings."""
    features = np.zeros((len(texts), embedding_dim))
    for i, text in enumerate(texts):
        words = text.split()
        word_embeddings = [embeddings.get(word) for word in words if word in embeddings]
        if word_embeddings:
            features[i] = np.mean(word_embeddings, axis=0)
    return features

def download_glove(glove_file):
    """Download and extract the GloVe embeddings file if it doesn't exist."""
    if not os.path.isfile(glove_file):
        logging.info(f"{glove_file} not found. Downloading...")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_file = "glove.6B.zip"

        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Ensure we got an OK response
        with open(zip_file, 'wb') as f:
            f.write(response.content)
        logging.info(f"Downloaded {zip_file}")

        # Extract the file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')
        logging.info(f"Extracted {zip_file} to get {glove_file}")
    else:
        logging.info(f"{glove_file} already exists.")

def evaluate_transformer_model(model, input_ids, attention_masks, labels):
    """Evaluate a transformer model."""
    # Put the model in evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Print the classification report
    print(classification_report(labels, predictions))

    return accuracy


def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return accuracy

def save_model(model, name):
    # Save the model to a file
    dump(model, os.path.join('models',name+'.joblib'))

def load_model(name):
    # Load the model from a file
    model = load(os.path.join('models', name+'.joblib'))
    return model

def save_vectorizer(vectorizer,name):
    logging.info('Saving vectorizer')
    with open('models/'+name+'.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)  

def load_vectorizer(name):
    logging.info('Loading vectorizer')
    with open('models/'+name+'.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

import torch
from transformers import DistilBertForSequenceClassification

def load_model_from_checkpoint():
    # Initialize the model with the same architecture as the one used during training
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2,  # Assuming binary classification (fake vs real)
        output_attentions=False,
        output_hidden_states=False,
    )

    # Load the state dictionary from the saved checkpoint
    model.load_state_dict(torch.load('./models/tranformer'))

    return model

import torch

def predict_transformer_model(model, input_ids, attention_mask):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    return predicted_class, probabilities