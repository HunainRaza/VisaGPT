from django.db import models
from django.contrib.auth.models import User


class ChatConversation(models.Model):
    """Chat conversation session"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=100, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Chat {self.session_id} - {self.user.username if self.user else 'Anonymous'}"
    
    class Meta:
        verbose_name = 'Chat Conversation'
        verbose_name_plural = 'Chat Conversations'
        ordering = ['-updated_at']


class ChatMessage(models.Model):
    """Individual chat message"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System')
    ]
    
    conversation = models.ForeignKey(
        ChatConversation, 
        on_delete=models.CASCADE, 
        related_name='messages'
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    context_used = models.TextField(blank=True, help_text="RAG context used for this response")
    
    class Meta:
        ordering = ['timestamp']
        verbose_name = 'Chat Message'
        verbose_name_plural = 'Chat Messages'
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class VisaKnowledgeBase(models.Model):
    """Knowledge base for RAG system"""
    country = models.CharField(max_length=50, db_index=True)
    visa_type = models.CharField(max_length=100, db_index=True)
    content = models.TextField()
    embedding = models.JSONField(default=list)
    source_url = models.URLField(blank=True)
    last_updated = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['country', 'visa_type']),
        ]
        verbose_name = 'Visa Knowledge'
        verbose_name_plural = 'Visa Knowledge Base'
    
    def __str__(self):
        return f"{self.country} - {self.visa_type}"
