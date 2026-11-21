# sentiment_app/views.py

from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .ml_model import predict_sentiment
from .models import SentimentPredictionLog

def sentiment_view(request):
    context = {}
    if request.method == 'POST':
        text_input = request.POST.get('financial_text', '')
        if text_input:
            result = predict_sentiment(text_input)
            sentiment_label = result['sentiment']
            score = result['score']

            context['result'] = sentiment_label
            context['score'] = score
            context['input_text'] = text_input
            
            # Optional: Save web UI predictions to database as well
            SentimentPredictionLog.objects.create(
                text=text_input,
                sentiment=sentiment_label,
                score=score
            )
            
    return render(request, 'sentiment_app/home.html', context)


# --- JSON API VIEW (Powered by Django REST Framework) ---
@api_view(['POST'])
def sentiment_api(request):
    """
    API endpoint that accepts JSON: {"text": "Your financial sentence here"}
    Returns JSON with sentiment prediction, score, and saves it to SQLite.
    """
    text_input = request.data.get('text', '')
    
    if not text_input:
        return Response({
            'status': 'error', 
            'message': 'No text provided in payload.'
        }, status=400)
    
    # Run prediction using your SimpleRNN model
    result = predict_sentiment(text_input)
    sentiment_label = result['sentiment']
    score = result['score']
    # Save the request and prediction result to db.sqlite3 table
    SentimentPredictionLog.objects.create(
        text=text_input,
        sentiment=sentiment_label,
        score=score
    )
    
    return Response({
        'status': 'success',
        'data': {
            'text': text_input,
            'sentiment': sentiment_label,
            'score': score
        }
    })