from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('predict/', views.PredictFormView.as_view(), name='predict_form'),
    path('predict/submit/', views.PredictView.as_view(), name='predict'),
    path('prediction/<int:pk>/', views.PredictionDetailView.as_view(), name='detail'),
]
