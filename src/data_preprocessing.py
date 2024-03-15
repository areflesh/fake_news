import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def load_data():
    # Load the data
    real_news = pd.read_csv('data/True.csv')
    fake_news = pd.read_csv('data/Fake.csv')

    # Label the data
    real_news['label'] = 'real'
    fake_news['label'] = 'fake'

    # Concatenate the dataframes
    df = pd.concat([real_news, fake_news])

    return df

def handle_missing_values(df):
    # Remove rows with NaN in 'text' column
    df = df.dropna(subset=['text'])

    return df

def preprocess_text(df):
    # Initialize the Porter Stemmer
    ps = PorterStemmer()

    # Initialize the list of stopwords
    stop_words = set(stopwords.words('english'))

    # Preprocess the text
    df['text'] = df['text'].apply(lambda x: ' '.join([ps.stem(word) for word in re.sub("[^a-zA-Z]", " ", x).lower().split() if word not in stop_words]))

    return df

def split_data(df):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test