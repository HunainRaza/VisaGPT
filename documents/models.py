from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    """Uploaded document for visa application"""
    DOCUMENT_TYPES = [
        ('passport', 'Passport'),
        ('diploma', 'Diploma/Degree'),
        ('transcript', 'Academic Transcript'),
        ('bank_statement', 'Bank Statement'),
        ('employment_letter', 'Employment Letter'),
        ('tax_return', 'Tax Return'),
        ('recommendation', 'Recommendation Letter'),
        ('other', 'Other'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPES)
    file = models.FileField(upload_to='documents/%Y/%m/')
    original_filename = models.CharField(max_length=255)
    file_size = models.IntegerField(help_text="File size in bytes")
    
    # OCR processing
    extracted_text = models.TextField(blank=True)
    extracted_data = models.JSONField(default=dict, help_text="Structured data extracted from document")
    processing_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Document'
        verbose_name_plural = 'Documents'
    
    def __str__(self):
        return f"{self.user.username} - {self.get_document_type_display()}"
    
    @property
    def is_processed(self):
        return self.processing_status == 'completed'


class DocumentValidation(models.Model):
    """Validation results for uploaded documents"""
    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='validation')
    is_valid = models.BooleanField(default=False)
    validation_issues = models.JSONField(default=list)
    completeness_score = models.FloatField(null=True, blank=True, help_text="0-100")
    validated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Document Validation'
        verbose_name_plural = 'Document Validations'
    
    def __str__(self):
        return f"Validation for {self.document}"
