from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('predict/', views.SelectCountryView.as_view(), name='predict_form'),
    path('predict/h1b/', views.H1BPredictFormView.as_view(), name='h1b_form'),
    path('predict/h1b/submit/', views.PredictView.as_view(), name='predict'),
    path('predict/canada/', views.CanadaCalculatorView.as_view(), name='canada_calculator'),
    path('prediction/<int:pk>/', views.PredictionDetailView.as_view(), name='detail'),
]
