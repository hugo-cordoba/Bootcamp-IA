import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import os

from apify_client import ApifyClient

# Initialize the ApifyClient with your Apify API token
client = ApifyClient("apify_api_tIlQhfH6kfksfM03YFxnGpZRr2PGpQ2vEvqi")

def train_model_sentiment():
    df = pd.read_csv('./datasets/Twitter_Data.csv')
    df_clean = df.dropna()
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df_clean['clean_text'])
    sequences = tokenizer.texts_to_sequences(df_clean['clean_text'])
    max_sequence_len = max([len(x) for x in sequences])
    X = pad_sequences(sequences, maxlen=max_sequence_len)
    y = to_categorical(np.asarray(df_clean['category'] + 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=max_sequence_len),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

    model.save('sentiment_model.keras')  # Guardar el modelo en formato .keras
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w') as file:
        file.write(tokenizer_json)  # Guardar el tokenizador

    return max_sequence_len, tokenizer, model

# Entrenar y guardar solo si es necesario
if not os.path.exists('sentiment_model.keras'):
    max_sequence_len, tokenizer, model = train_model_sentiment()
else:
    model = load_model('sentiment_model.keras')
    with open('tokenizer.json', 'r') as file:
        tokenizer_json = file.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        max_sequence_len = 100

def load_comments(instagram_url):
    comments = []
    run_input = {
        "directUrls": [instagram_url],
        "resultsType": "posts",
        "resultsLimit": 200,
        "searchType": "hashtag",
        "searchLimit": 1,
    }
    run = client.actor("apify/instagram-scraper").call(run_input=run_input)

    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        for dict_comments in item['latestComments']:
            text = dict_comments.get('text')
            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=max_sequence_len)
            pred = model.predict(padded)
            predicted_category_index = np.argmax(pred[0])
            categories = {-1: 'Negativo', 0: 'Neutral', 1: 'Positivo'}
            predicted_category = categories[predicted_category_index - 1]
            
            comments.append({
                'username': dict_comments.get('ownerUsername'),
                'comment': text,
                'sentiment': predicted_category
            })
    return comments
