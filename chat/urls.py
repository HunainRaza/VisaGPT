from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('chat/', views.ChatView.as_view(), name='chat'),
    path('api/message/', views.ChatMessageView.as_view(), name='chat_message'),
    path('new/', views.NewChatView.as_view(), name='new_chat'),
]
