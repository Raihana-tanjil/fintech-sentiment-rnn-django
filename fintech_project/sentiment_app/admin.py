from django.contrib import admin
from .models import SentimentPredictionLog

@admin.register(SentimentPredictionLog)
class SentimentPredictionLogAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'sentiment', 'score', 'text')
    list_filter = ('sentiment', 'created_at')
    search_fields = ('text',)