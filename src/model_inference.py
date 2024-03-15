import argparse
import logging
import pandas as pd
from data_preprocessing import preprocess_text, handle_missing_values
from utils import load_vectorizer
from utils import load_glove_vectors, load_model_from_checkpoint, load_model, predict_transformer_model
from feature_engineering import glove_features_generation, create_features_transformer_inference

def load_and_preprocess_data(input_file):

    df = pd.read_csv(input_file)
    logging.info("Data loaded")
    df = handle_missing_values(df)
    df = preprocess_text(df)
    logging.info("Text preprocessed")

    return df

def main(input_file, output_file, model_type, feature_engineering):
 
    df = load_and_preprocess_data(input_file)
    vectorizers = {'tfidf': load_vectorizer('tfidf'), 'bow': load_vectorizer('bow')}
    
    if args.feature_engineering in vectorizers.keys():
        vectorizer = vectorizers[args.feature_engineering]
        X_features = vectorizer.transform(df['text'])
    elif args.feature_engineering == 'glove':
        embeddings = load_glove_vectors('glove.6B.50d.txt')
        X_features  = glove_features_generation(df['text'], embeddings, 50)
    elif args.feature_engineering == 'transformer':
        X_ids, X_masks, _ = create_features_transformer_inference(df['text'])
        X_features = X_ids, X_masks    
    
    

    if feature_engineering != 'transformer':
        model = load_model(model_type)
        predictions = model.predict(X_features)

    else:
        try:
            model = load_model_from_checkpoint()
        except:
            logging.info("No tranformer checkpoint found")
        predictions, _ = predict_transformer_model(model, X_ids, X_masks)

    # Save predictions
    df['predictions'] = predictions
    df.to_csv(output_file, index=False)
    logging.info("Predictions saved to {}".format(output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform inference using trained models')
    parser.add_argument('--input_file', type=str, required=True, help='Input file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path')
    parser.add_argument('--model_file', type=str, required=True, help='Model file path')
    parser.add_argument('--feature_engineering', choices=['tfidf', 'bow', 'glove', 'transformer'], required=True, help='Type of feature engineering to use')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args.input_file, args.output_file, args.model_file, args.feature_engineering)
