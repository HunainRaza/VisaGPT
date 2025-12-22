from django.db import models
from django.contrib.auth.models import User


class VisaPrediction(models.Model):
    """ML-based visa approval prediction"""
    VISA_TYPE_CHOICES = [
        ('H-1B', 'H-1B (Work Visa)'),
        ('F-1', 'F-1 (Student Visa)'),
        ('L-1', 'L-1 (Intra-company Transfer)'),
        ('O-1', 'O-1 (Extraordinary Ability)'),
        ('B-1/B-2', 'B-1/B-2 (Tourist/Business)'),
    ]
    
    COUNTRY_CHOICES = [
        ('USA', 'United States'),
        ('Canada', 'Canada'),
        ('UK', 'United Kingdom'),
        ('Australia', 'Australia'),
        ('Germany', 'Germany'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    visa_type = models.CharField(max_length=50, choices=VISA_TYPE_CHOICES)
    country = models.CharField(max_length=50, choices=COUNTRY_CHOICES)
    
    # Applicant details
    age = models.IntegerField(null=True, blank=True)
    education_level = models.CharField(max_length=50, blank=True)
    work_experience_years = models.IntegerField(null=True, blank=True)
    salary_offered = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    language_score = models.FloatField(null=True, blank=True, help_text="IELTS/TOEFL score")
    job_title = models.CharField(max_length=200, blank=True)
    employer_name = models.CharField(max_length=200, blank=True)
    
    # Prediction results
    approval_probability = models.FloatField(help_text="Probability between 0 and 1")
    confidence_score = models.FloatField(null=True, blank=True)
    risk_factors = models.JSONField(default=list, help_text="List of potential issues")
    recommendations = models.TextField(blank=True)
    model_version = models.CharField(max_length=50, default='1.0')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Visa Prediction'
        verbose_name_plural = 'Visa Predictions'
    
    def __str__(self):
        return f"{self.user.username} - {self.visa_type} ({self.approval_probability:.0%})"
    
    @property
    def approval_percentage(self):
        """Return approval probability as percentage"""
        return self.approval_probability * 100
    
    @property
    def risk_level(self):
        """Categorize risk based on probability"""
        if self.approval_probability >= 0.8:
            return 'Low'
        elif self.approval_probability >= 0.6:
            return 'Medium'
        else:
            return 'High'
