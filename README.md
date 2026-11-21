# Fintech Sentiment Analysis (SimpleRNN + Django)

Binary fintech sentiment analysis (`Bullish / Positive` vs. `Bearish / Negative`) using an `Embedding` + `SimpleRNN(64)` model, served through a Django web interface and a Django REST Framework `/api/predict/` endpoint with SQLite logging (`SentimentPredictionLog`).

## Project Structure
- `fintech_sentiment_rnn.py` - Trains the `SimpleRNN` model and exports `fintech_rnn_model.keras` and `word_index.pkl` to `fintech_project/sentiment_app/`.
- `fintech_project/` - Django project and `sentiment_app` (web UI, `/api/predict/` DRF endpoint, `SentimentPredictionLog` model, admin registration, and `test_api.py`).

## Setup and Run
1. Install dependencies: `pip install tensorflow numpy pandas scikit-learn django djangorestframework requests`
2. Run migrations inside `fintech_project`:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```
3. Start the server:
   ```bash
   python manage.py runserver
   ```
4. In a second terminal, run API tests:
   ```bash
   python sentiment_app/test_api.py
   ```
