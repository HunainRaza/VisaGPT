from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.generic import ListView, DetailView, TemplateView, View
from .models import VisaPrediction
from ml_engine.train_model import VisaMLModel
import os

# Load ML model
ml_model = VisaMLModel()
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_models')

try:
    ml_model.load_model(model_path)
    print("✓ ML model loaded successfully")
except Exception as e:
    print(f"⚠ ML model not loaded: {e}")


@method_decorator(login_required, name='dispatch')
class DashboardView(ListView):
    """User dashboard showing all predictions"""
    model = VisaPrediction
    template_name = 'predictions/dashboard.html'
    context_object_name = 'predictions'
    
    def get_queryset(self):
        return VisaPrediction.objects.filter(user=self.request.user)[:10]
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Calculate statistics
        all_predictions = VisaPrediction.objects.filter(user=self.request.user)
        context['total_predictions'] = all_predictions.count()
        context['high_confidence'] = all_predictions.filter(approval_probability__gte=0.8).count()
        
        return context


@method_decorator(login_required, name='dispatch')
class PredictFormView(TemplateView):
    """Display prediction form"""
    template_name = 'predictions/predict_form.html'


@method_decorator(login_required, name='dispatch')
class PredictView(View):
    """Process prediction request"""
    
    def post(self, request):
        # Get form data
        visa_type = request.POST.get('visa_type')
        country = request.POST.get('country')
        job_title = request.POST.get('job_title', '')
        salary = float(request.POST.get('salary', 0))
        age = int(request.POST.get('age', 0))
        education = request.POST.get('education', '')
        experience = int(request.POST.get('experience', 0))
        
        # Prepare features for ML model
        features = {
            'IS_TECH_JOB': 1 if any(word in job_title.lower() for word in ['software', 'engineer', 'developer', 'data', 'analyst']) else 0,
            'WAGE_LEVEL': min(5, max(1, int(salary / 20000))),
            'IS_FULL_TIME': 1,
            'EMPLOYER_FREQUENCY': 50,
            'STATE_ENCODED': 0,
            'SOC_ENCODED': 0
        }
        
        # Get prediction from ML model
        try:
            probability = ml_model.predict(features)
        except Exception as e:
            print(f"ML prediction error: {e}")
            probability = 0.75  # Fallback
        
        # Generate recommendations based on probability
        if probability >= 0.8:
            recommendation = """Strong application! Your profile shows high approval likelihood.
            Focus on: Complete documentation, accurate form filling, timely submission."""
            risk_factors = []
        elif probability >= 0.6:
            recommendation = """Good chances of approval. Consider strengthening:
            - Highlight unique qualifications
            - Ensure all documents are properly certified
            - Consider professional review of application"""
            risk_factors = ["Moderate competition", "Documentation completeness"]
        else:
            recommendation = """Application may need strengthening. Suggestions:
            - Improve qualifications (education, experience)
            - Consider alternative visa types
            - Consult with immigration attorney
            - Wait for better timing/qualifications"""
            risk_factors = ["Low salary range", "Limited experience", "High competition"]
        
        # Save prediction
        prediction = VisaPrediction.objects.create(
            user=request.user,
            visa_type=visa_type,
            country=country,
            job_title=job_title,
            salary_offered=salary,
            age=age,
            education_level=education,
            work_experience_years=experience,
            approval_probability=probability,
            recommendations=recommendation,
            risk_factors=risk_factors
        )
        
        return render(request, 'predictions/result.html', {
            'prediction': prediction
        })
    
    def get(self, request):
        # Redirect to form if accessed via GET
        return redirect('predictions:predict_form')


@method_decorator(login_required, name='dispatch')
class PredictionDetailView(DetailView):
    """View individual prediction details"""
    model = VisaPrediction
    template_name = 'predictions/details.html'
    context_object_name = 'prediction'
    
    def get_queryset(self):
        # Only allow users to view their own predictions
        return VisaPrediction.objects.filter(user=self.request.user)


# Backward compatibility - keep old names as aliases
dashboard = DashboardView.as_view()
predict_form = PredictFormView.as_view()
predict = PredictView.as_view()
prediction_detail = PredictionDetailView.as_view()