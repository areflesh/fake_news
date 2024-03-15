from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import requests
import numpy as np
import zipfile
import logging
import os
from transformers import DistilBertTokenizer 
import torch
from utils import load_glove_vectors, glove_features_generation
from utils import save_vectorizer


logging.basicConfig(level=logging.INFO)



def create_features_glove(X_train, X_test):
    embeddings = load_glove_vectors('glove.6B.50d.txt')
    X_train_glove = glove_features_generation(X_train, embeddings, 50)
    X_test_glove = glove_features_generation(X_test, embeddings, 50)

    return X_train_glove, X_test_glove

def create_features_tfidf(X_train, X_test):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    save_vectorizer(tfidf, 'tfidf')

    return X_train_tfidf, X_test_tfidf

def create_features_bow(X_train, X_test):
    count = CountVectorizer()
    X_train_bow = count.fit_transform(X_train)
    X_test_bow = count.transform(X_test)
    save_vectorizer(count, 'bow')

    return X_train_bow, X_test_bow

def create_features_transformer(X, y):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    max_length = 128
    input_ids = []
    attention_masks = []
    labels = []

    for text, label in zip(X, y):
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(torch.tensor([1 if label == 'real' else 0]))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

def create_features_transformer_inference(X):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    max_length = 128
    input_ids = []
    attention_masks = []
    labels = []

    for text in X:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    

    return input_ids, attention_masks