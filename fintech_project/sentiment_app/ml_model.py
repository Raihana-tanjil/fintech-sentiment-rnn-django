# sentiment_app/ml_model.py
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Define paths relative to the app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'fintech_rnn_model.keras')
INDEX_PATH = os.path.join(BASE_DIR, 'word_index.pkl')

# Load model and word index globally so they stay in memory
model = load_model(MODEL_PATH)
with open(INDEX_PATH, 'rb') as f:
    word_index = pickle.load(f)

max_length = 100

def pad_sequences_custom(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        truncated = seq[:maxlen]
        padded[i, :len(truncated)] = truncated
    return padded

def predict_sentiment(text):
    words = text.lower().split()
    seq = [word_index.get(w, 1) for w in words]
    padded = pad_sequences_custom([seq], max_length)
    
    prediction = float(model.predict(padded)[0][0])
    sentiment = "Bullish / Positive" if prediction > 0.5 else "Bearish / Negative"
    
    return {
        'text': text,
        'sentiment': sentiment,
        'score': round(prediction, 4)
    }