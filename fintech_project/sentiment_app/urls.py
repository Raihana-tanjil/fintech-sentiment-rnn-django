from django.urls import path
from . import views

urlpatterns = [
    path('', views.sentiment_view, name='home'),             # HTML web page view
    path('api/predict/', views.sentiment_api, name='sentiment_api'), # JSON API endpoint view
]