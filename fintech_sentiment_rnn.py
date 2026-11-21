from builtins import print

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

# --- CONFIGURATION ---
vocab_size = 10000  
max_length = 100    
embedding_dim = 64  
batch_size = 32
epochs = 5

print("1. Loading and inspecting the Fintech dataset...")
try:
    df = pd.read_csv('fintech_data.csv')
except FileNotFoundError:
    print("Error: 'fintech_data.csv' not found. Using fallback dummy fintech data.")
    data = {
        'text': [
            "Company reports record profits and soaring quarterly revenue growth.",
            "Stock plunges following unexpected regulatory investigation and losses.",
            "Market remains stable with modest gains across banking sector.",
            "Severe risk of default as debt levels rise significantly.",
            "Strong bullish outlook driven by innovative fintech product launch."
        ],
        'label': [1, 0, 1, 0, 1] # 1 = Positive, 0 = Negative
    }
    df = pd.DataFrame(data)

df = df.dropna(subset=['text', 'label'])
texts = df['text'].values
labels = df['label'].values

print(f"Total samples loaded: {len(texts)}")

print("\n2. Tokenizing and Padding Text Data (Native Python)...")
# Simple native tokenizer to map words to integers
word_index = {"<OOV>": 1}
def tokenize_text(corpus, vocab_limit):
    sequences = []
    for text in corpus:
        words = text.lower().split()
        seq = []
        for word in words:
            if word not in word_index and len(word_index) < vocab_limit:
                word_index[word] = len(word_index) + 1
            seq.append(word_index.get(word, 1))
        sequences.append(seq)
    return sequences

raw_sequences = tokenize_text(texts, vocab_size)

# Simple native padding function
def pad_sequences_custom(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        truncated = seq[:maxlen]
        padded[i, :len(truncated)] = truncated
    return padded

X = pad_sequences_custom(raw_sequences, max_length)
y = np.array(labels)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n3. Building the SimpleRNN model architecture...")
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    SimpleRNN(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

model.summary()

print("\n4. Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test)
)

print("\n5. Evaluating model performance...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

# --- INFERENCE FUNCTION ---
def predict_financial_sentiment(sentence):
    words = sentence.lower().split()
    seq = [word_index.get(w, 1) for w in words]
    padded = pad_sequences_custom([seq], max_length)
    prediction = model.predict(padded)[0][0]
    
    sentiment = "Bullish / Positive" if prediction > 0.5 else "Bearish / Negative"
    return f"Text: '{sentence}'\nPrediction: {sentiment} (Score: {prediction:.4f})\n"

print("\n--- Test Predictions ---")
print(predict_financial_sentiment("The startup secured a massive funding round with high revenue potential."))
print(predict_financial_sentiment("Shares dropped sharply after missing quarterly earnings expectations."))


import os
import pickle

# Define the exact destination folder inside your Django project
django_app_dir = r"F:\NLP\Fintech_2.0\fintech_project\sentiment_app"

# Ensure the directory exists
os.makedirs(django_app_dir, exist_ok=True)

# Save the trained model and word index directly into Django's folder
model_path = os.path.join(django_app_dir, "fintech_rnn_model.keras")
index_path = os.path.join(django_app_dir, "word_index.pkl")

model.save(model_path)
with open(index_path, "wb") as f:
    pickle.dump(word_index, f)

print(f"Model and word index saved successfully inside: {django_app_dir}")

