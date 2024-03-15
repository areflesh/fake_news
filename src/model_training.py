import argparse
import logging
from data_preprocessing import load_data, handle_missing_values, preprocess_text, split_data
from feature_engineering import create_features_tfidf, create_features_bow, create_features_glove, create_features_transformer
from models import train_model_mlp, train_model_random_forest, train_model_transformer
from utils import evaluate_model, evaluate_transformer_model, save_model

def train_models(args):
    df = load_data()
    logging.info("Data loaded")
    df = handle_missing_values(df)
    logging.info("Missing values handled")
    df = preprocess_text(df)
    logging.info("Text preprocessed")
    X_train, X_test, y_train, y_test = split_data(df)

    feature_engineering = {
        'tfidf': create_features_tfidf,
        'bow': create_features_bow,
        'glove': create_features_glove}

    model_types = {
        'mlp': train_model_mlp,
        'random_forest': train_model_random_forest,
        'transformer': train_model_transformer}

    for model_type in args.models:
        if model_type != 'transformer':

            for name, func in feature_engineering.items():
                X_train_f, X_test_f = func(X_train, X_test)

                logging.info("Features {} created. Model training started".format(name))
                model = model_types[model_type](X_train_f, y_train)
                
                accuracy = evaluate_model(model, X_test_f, y_test)
                logging.info("Model {} trained and evaluated".format(model_type))
                print(f'Accuracy: {accuracy}')
                
                save_model(model, f'{name}_{model_type}_model')
        else:
           
            X_train_ids, X_train_masks, y_train_tensors = create_features_transformer(X_train, y_train)
            X_test_ids, X_test_masks, y_test_tensors = create_features_transformer(X_test, y_test)
            
            logging.info("Features for transformer created. Model training started")
            model = train_model_transformer(X_train_ids, X_train_masks, y_train_tensors, epochs=4, batch_size=300)
            
            evaluate_transformer_model(model, X_test_ids, X_test_masks, y_test_tensors)
            logging.info("Transformer model trained and evaluated")
                

def main():
    parser = argparse.ArgumentParser(description='Train different models with various feature engineering techniques')
    parser.add_argument('--models', nargs='+', choices=['mlp', 'random_forest', 'transformer'], help='Types of models to train')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_models(args)

if __name__ == "__main__":
    main()


