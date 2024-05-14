import numpy as np
import pandas as pd
import emoji
import json
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from apify_client import ApifyClient

# Initialize the ApifyClient with your Apify API token
client = ApifyClient("apify_api_tIlQhfH6kfksfM03YFxnGpZRr2PGpQ2vEvqi")

def train_model_sentiment(dataset_path, model_name, tokenizer_name, max_len_name, vocab_size=5000):
    df = pd.read_csv(dataset_path)
    df_clean = df.dropna()
    
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df_clean['clean_text'])
    sequences = tokenizer.texts_to_sequences(df_clean['clean_text'])
    max_sequence_len = max(len(x) for x in sequences)
    X = pad_sequences(sequences, maxlen=max_sequence_len)
    y = to_categorical(np.asarray(df_clean['category'] + 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_sequence_len),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

    model.save(f'{model_name}.keras')  # Guardar el modelo en formato .keras
    tokenizer_json = tokenizer.to_json()
    with open(f'{tokenizer_name}.json', 'w') as file:
        file.write(tokenizer_json)  # Guardar el tokenizador
        
    # Guardar max_sequence_len
    with open(f'{max_len_name}.json', 'w') as file:
        json.dump({'max_sequence_len': max_sequence_len}, file)

    return max_sequence_len, tokenizer, model

def load_or_train_model(dataset_path, model_name, tokenizer_name, max_len_name):
    if os.path.exists(f'{model_name}.keras'):
        model = load_model(f'{model_name}.keras')
        with open(f'{tokenizer_name}.json', 'r') as file:
            tokenizer = tokenizer_from_json(file.read())
        with open(f'{max_len_name}.json', 'r') as file:
            max_sequence_len = json.load(file)['max_sequence_len']
    else:
        max_sequence_len, tokenizer, model = train_model_sentiment(
            dataset_path, model_name, tokenizer_name, max_len_name)
    return max_sequence_len, tokenizer, model

# Cargar o entrenar los modelos
max_sequence_len_1, tokenizer_1, model_1 = load_or_train_model(
    './datasets/combined_text_emoji_sentiment_dataset.csv',
    'sentiment_model_1', 'tokenizer_1', 'max_sequence_len_1'
)

max_sequence_len_2, tokenizer_2, model_2 = load_or_train_model(
    './datasets/emoji_sentiment_dataset.csv',
    'sentiment_model_2', 'tokenizer_2', 'max_sequence_len_2'
)

def emoji_to_unicode_name(em):
    return emoji.demojize(em)

def process_comments(comments, model, tokenizer, max_sequence_len):
    sentiment_count = {'Positivo': 0, 'Neutral': 0, 'Negativo': 0}
    processed_comments = []

    for comment in comments:
        text = comment.get('text')
        text_unicode_names = pd.Series([text]).apply(emoji_to_unicode_name).str.replace(':', ' ').tolist()
        seq = tokenizer.texts_to_sequences(text_unicode_names)
        padded = pad_sequences(seq, maxlen=max_sequence_len)
        pred = model.predict(padded)
        
        predicted_category_index = np.argmax(pred[0])
        categories = {-1: 'Negativo', 0: 'Neutral', 1: 'Positivo'}
        predicted_category = categories[predicted_category_index - 1]

        processed_comments.append({
            'username': comment.get('ownerUsername'),
            'comment': text,
            'sentiment': predicted_category
        })
        sentiment_count[predicted_category] += 1
    
    return processed_comments, sentiment_count

def load_comments(instagram_url):
    run_input = {
        "directUrls": [instagram_url],
        "resultsType": "posts",
        "resultsLimit": 200,
        "searchType": "hashtag",
        "searchLimit": 1,
    }
    run = client.actor("apify/instagram-scraper").call(run_input=run_input)

    all_comments = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        all_comments.extend(item['latestComments'])
    
    # comments, sentiment_count = process_comments(all_comments, model_1, tokenizer_1, max_sequence_len_1)
    comments, sentiment_count = process_comments(all_comments, model_2, tokenizer_2, max_sequence_len_2)
    
    print(all_comments)
    
    return comments, sentiment_count

def calculate_percentage(sentiment_count):
    total_comments = sum(sentiment_count.values())
    percentages = {key: (value / total_comments) * 100 for key, value in sentiment_count.items()}
    most_frequent = max(percentages, key=percentages.get)
    return percentages, most_frequent
