"""
Predictions Views - CORRECTED FEATURE ENGINEERING
Feature values now match training data distribution
"""
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.generic import ListView, DetailView, TemplateView, View
from .models import VisaPrediction
from ml_engine.train_model import VisaMLModel
import os
import numpy as np


# Load ML model
ml_model = VisaMLModel()
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_models')

try:
    ml_model.load_model(model_path)
    print("✓ ML model loaded successfully")
    print(f"  Features: {ml_model.feature_columns}")
except Exception as e:
    print(f"⚠️ ML model not loaded: {e}")


@method_decorator(login_required, name='dispatch')
class DashboardView(ListView):
    """User dashboard showing all predictions"""
    model = VisaPrediction
    template_name = 'predictions/dashboard.html'
    context_object_name = 'predictions'
    
    def get_queryset(self):
        return VisaPrediction.objects.filter(user=self.request.user).order_by('-created_at')[:10]
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        all_predictions = VisaPrediction.objects.filter(user=self.request.user)
        context['total_predictions'] = all_predictions.count()
        context['high_confidence'] = all_predictions.filter(approval_probability__gte=0.8).count()
        context['medium_confidence'] = all_predictions.filter(
            approval_probability__gte=0.6, 
            approval_probability__lt=0.8
        ).count()
        context['low_confidence'] = all_predictions.filter(approval_probability__lt=0.6).count()
        return context


@method_decorator(login_required, name='dispatch')
class SelectCountryView(TemplateView):
    """Landing page to select country/visa type"""
    template_name = 'predictions/select_country.html'

@method_decorator(login_required, name='dispatch')
class H1BPredictFormView(TemplateView):
    """Display H-1B prediction form"""
    template_name = 'predictions/predict_form.html'


@method_decorator(login_required, name='dispatch')
class PredictView(View):
    """Process prediction request with CORRECTED feature engineering"""
    
    def post(self, request):
        # ============================================
        # STEP 1: COLLECT FORM DATA
        # ============================================
        
        visa_type = request.POST.get('visa_type', 'H-1B')
        country = request.POST.get('country', 'USA')
        
        # Job information
        job_title = request.POST.get('job_title', '').strip()
        salary = float(request.POST.get('salary', 0))
        employment_type = request.POST.get('employment_type', 'full-time')
        employer_size = request.POST.get('employer_size', 'medium')
        location = request.POST.get('location', 'Other')
        
        # Personal information
        age = int(request.POST.get('age', 30))
        education = request.POST.get('education', 'Bachelors')
        experience = int(request.POST.get('experience', 0))
        degree_relevance = request.POST.get('degree_relevance', 'related')
        previous_h1b = request.POST.get('previous_h1b', 'no')
        
        # ============================================
        # STEP 2: FEATURE ENGINEERING (CORRECTED!)
        # ============================================
        
        # Feature 1: WAGE_LEVEL (1-5)
        # Training bins: [0, 40k, 60k, 80k, 100k, 500k]
        if salary < 40000:
            wage_level = 1.0
        elif salary < 60000:
            wage_level = 2.0
        elif salary < 80000:
            wage_level = 3.0
        elif salary < 100000:
            wage_level = 4.0
        else:
            wage_level = 5.0
        
        # Feature 2: IS_TECH_JOB (0 or 1)
        tech_keywords = [
            'software', 'engineer', 'developer', 'programmer',
            'analyst', 'data', 'scientist', 'architect', 'devops',
            'machine learning', 'ai', 'artificial intelligence',
            'technology', 'computer', 'it', 'tech', 'web'
        ]
        is_tech_job = 1.0 if any(keyword in job_title.lower() for keyword in tech_keywords) else 0.0
        
        # Feature 3: IS_FULL_TIME (0 or 1)
        is_full_time = 1.0 if employment_type == 'full-time' else 0.0
        
        # Feature 4: EMPLOYER_FREQUENCY (log scale)
        # CRITICAL FIX: Match training distribution (0.69 to 9.48, mean 3.58)
        # Training: log1p(number of H-1B filings by that employer)
        # Startup that never filed = 1 filing → log1p(1) = 0.69
        # Fortune 500 = 10,000 filings → log1p(10000) = 9.21
        employer_count_map = {
            'startup': 1,        # log1p(1) = 0.69 (MINIMUM - never filed)
            'small': 10,         # log1p(10) = 2.40 (filed a few times)
            'medium': 100,       # log1p(100) = 4.62 (regular filer)
            'large': 1000,       # log1p(1000) = 6.91 (major filer)
            'fortune500': 10000  # log1p(10000) = 9.21 (MAXIMUM - top filer)
        }
        employer_count = employer_count_map.get(employer_size, 30)  # Default to ~3.47
        employer_frequency = np.log1p(employer_count)
        
        # Salary-based adjustment for employer frequency
        # High salary often correlates with big companies in real data
        if salary >= 120000:
            # Very high salary → likely top-tier employer
            employer_frequency = min(9.48, employer_frequency * 1.25)
        elif salary >= 100000:
            employer_frequency = min(9.48, employer_frequency * 1.15)
        elif salary >= 85000:
            employer_frequency = min(9.48, employer_frequency * 1.08)
        
        # Feature 5: STATE_ENCODED
        # CRITICAL FIX: Approximate training distribution
        # Training had states encoded 0-52 by LabelEncoder
        # We approximate based on H-1B popularity:
        # - High activity states (CA, NY, TX, WA, MA) → Higher numbers
        # - Medium activity → Middle numbers  
        # - Low activity → Lower numbers
        # This isn't perfect but much better than arbitrary values
        state_encoding_map = {
            # Top tech hubs (high approval, high activity)
            'California': 45,        # Very high
            'Washington': 47,        # Seattle tech hub
            'Massachusetts': 42,     # Boston tech/education
            
            # Major business hubs
            'New York': 38,          # Finance/tech
            'Texas': 35,             # Growing tech
            'Illinois': 32,          # Chicago business
            
            # Mid-tier states
            'New Jersey': 28,        # Pharma/tech
            'Virginia': 25,          # DC area tech
            'Florida': 22,           # Growing market
            'Pennsylvania': 20,      # Mid-size market
            
            # Other states (lower activity)
            'Other': 10              # Below average
        }
        state_encoded = float(state_encoding_map.get(location, 10))
        
        # Feature 6: SOC_ENCODED
        # CRITICAL FIX: Better approximation of occupation codes
        # Training had top 100 occupations + OTHER encoded 0-100
        # We approximate based on occupation type
        if is_tech_job == 1.0:
            # Tech occupations - these were most common in training
            if 'software' in job_title.lower() or 'developer' in job_title.lower():
                soc_encoded = 5.0   # Top tech occupation
            elif 'data' in job_title.lower() or 'scientist' in job_title.lower():
                soc_encoded = 8.0   # Data science occupations
            elif 'engineer' in job_title.lower():
                soc_encoded = 10.0  # Engineering occupations
            else:
                soc_encoded = 15.0  # Other tech
        elif any(word in job_title.lower() for word in ['manager', 'director', 'executive', 'lead']):
            soc_encoded = 25.0  # Management occupations
        elif any(word in job_title.lower() for word in ['analyst', 'consultant', 'specialist']):
            soc_encoded = 35.0  # Professional services
        elif any(word in job_title.lower() for word in ['accountant', 'financial', 'finance']):
            soc_encoded = 45.0  # Business/finance
        elif any(word in job_title.lower() for word in ['teacher', 'professor', 'instructor', 'educator']):
            soc_encoded = 55.0  # Education
        elif any(word in job_title.lower() for word in ['designer', 'artist', 'creative']):
            soc_encoded = 65.0  # Arts/design
        elif any(word in job_title.lower() for word in ['sales', 'marketing']):
            soc_encoded = 75.0  # Sales/marketing
        elif any(word in job_title.lower() for word in ['nurse', 'medical', 'health']):
            soc_encoded = 50.0  # Healthcare
        else:
            soc_encoded = 85.0  # Other occupations (lower approval)
        
        # ============================================
        # ADJUSTMENT FACTORS (Applied after base prediction)
        # ============================================
        
        # Education bonus
        education_multiplier = {
            'PhD': 1.05,        # +5%
            'Masters': 1.02,    # +2%
            'Bachelors': 1.0    # Baseline
        }.get(education, 1.0)
        
        # Experience bonus
        if experience >= 7:
            experience_bonus = 1.03
        elif experience >= 5:
            experience_bonus = 1.02
        elif experience >= 3:
            experience_bonus = 1.01
        elif experience == 0:
            experience_bonus = 0.95  # -5% for no experience
        else:
            experience_bonus = 1.0
        
        # Degree relevance
        degree_penalty = {
            'exact': 1.0,       # Perfect match
            'related': 0.98,    # -2%
            'unrelated': 0.90   # -10% (increased penalty)
        }.get(degree_relevance, 1.0)
        
        # Previous H-1B bonus
        h1b_bonus = {
            'transfer': 1.05,   # +5%
            'yes': 1.03,        # +3%
            'no': 1.0
        }.get(previous_h1b, 1.0)
        
        # Age penalty
        if age >= 50:
            age_penalty = 0.90   # -10%
        elif age >= 45:
            age_penalty = 0.94   # -6%
        elif age >= 40:
            age_penalty = 0.97   # -3%
        else:
            age_penalty = 1.0
        
        # ============================================
        # STEP 3: CREATE FEATURES DICTIONARY
        # ============================================
        
        features = {
            'WAGE_LEVEL': wage_level,
            'IS_TECH_JOB': is_tech_job,
            'IS_FULL_TIME': is_full_time,
            'EMPLOYER_FREQUENCY': employer_frequency,
            'STATE_ENCODED': state_encoded,
            'SOC_ENCODED': soc_encoded
        }
        
        print(f"\n{'='*70}")
        print(f"PREDICTION REQUEST - CORRECTED FEATURES")
        print(f"{'='*70}")
        print(f"Input Data:")
        print(f"  Job: {job_title}")
        print(f"  Salary: ${salary:,}")
        print(f"  Employment: {employment_type}")
        print(f"  Employer: {employer_size}")
        print(f"  Location: {location}")
        print(f"  Age: {age}, Education: {education}, Experience: {experience} years")
        print(f"\nCalculated Features:")
        for k, v in features.items():
            print(f"  {k}: {v:.2f}")
        print(f"\nTraining Data Reference:")
        print(f"  EMPLOYER_FREQUENCY: 0.69 to 9.48 (mean 3.58)")
        print(f"    - Startup (1 filing): 0.69")
        print(f"    - Small (10 filings): 2.40")
        print(f"    - Medium (100 filings): 4.62")
        print(f"    - Large (1000 filings): 6.91")
        print(f"    - Fortune 500 (10000 filings): 9.21")
        print(f"  STATE_ENCODED: 0 to 52")
        print(f"  SOC_ENCODED: 0 to 100")
        
        # ============================================
        # STEP 4: GET ML MODEL PREDICTION
        # ============================================
        
        try:
            # Get base probability from model
            base_probability = ml_model.predict(features)
            
            # Apply adjustment factors
            adjusted_probability = (
                base_probability * 
                education_multiplier * 
                experience_bonus * 
                degree_penalty * 
                h1b_bonus * 
                age_penalty
            )
            
            # Ensure probability stays in valid range
            probability = min(0.99, max(0.01, adjusted_probability))
            
            print(f"\nPrediction Results:")
            print(f"  Base ML prediction: {base_probability:.1%}")
            print(f"  Education bonus: {education_multiplier}")
            print(f"  Experience bonus: {experience_bonus}")
            print(f"  Degree penalty: {degree_penalty}")
            print(f"  H-1B bonus: {h1b_bonus}")
            print(f"  Age penalty: {age_penalty}")
            print(f"  FINAL: {probability:.1%}")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            import traceback
            traceback.print_exc()
            probability = 0.70  # Fallback
        
        # ============================================
        # STEP 5: GENERATE RECOMMENDATIONS
        # ============================================
        
        risk_factors = []
        strengths = []
        
        # Analyze factors
        if salary < 50000:
            risk_factors.append("Salary below typical H-1B range ($50k+)")
        elif salary >= 90000:
            strengths.append("Competitive salary well above prevailing wage")
        
        if employment_type != 'full-time':
            risk_factors.append("Part-time position (H-1B requires full-time)")
        
        if is_tech_job == 0.0:
            risk_factors.append("Non-tech occupation faces higher scrutiny")
        else:
            strengths.append("Tech/STEM occupation (high demand)")
        
        if employer_size in ['startup', 'small']:
            risk_factors.append("Small employer (limited H-1B experience)")
        elif employer_size in ['large', 'fortune500']:
            strengths.append("Large established employer with H-1B track record")
        
        if education == 'PhD':
            strengths.append("PhD degree (advanced degree cap exempt)")
        elif education == 'Masters':
            strengths.append("Master's degree")
        
        if experience >= 5:
            strengths.append("Significant relevant work experience")
        elif experience == 0:
            risk_factors.append("No prior work experience")
        
        if degree_relevance == 'unrelated':
            risk_factors.append("Degree not directly related to job (specialty occupation requirement)")
        elif degree_relevance == 'exact':
            strengths.append("Degree directly matches job specialty")
        
        if previous_h1b in ['yes', 'transfer']:
            strengths.append("Previous H-1B experience")
        
        if age >= 45:
            risk_factors.append("Age may affect employer ROI considerations")
        
        # Generate recommendation text
        if probability >= 0.8:
            recommendation = f"""🟢 <strong>Strong Application Profile</strong>

        Your profile shows a <strong>high likelihood of approval ({probability:.0%})</strong>. 

        <strong>Key Strengths:</strong>
        {chr(10).join(['• ' + s for s in strengths]) if strengths else '• Strong overall profile'}

        <strong>Next Steps:</strong>
        • Ensure all documentation is complete and accurate
        • File petition during April cap season
        • Consider premium processing for faster results
        • Maintain consistent employment status"""
        elif probability >= 0.6:
            recommendation = f"""🟡 <strong>Moderate Approval Chances</strong>

        Your profile has a <strong>reasonable chance of approval ({probability:.0%})</strong>, but there are areas that could be strengthened.

        <strong>Strengths:</strong>
        {chr(10).join(['• ' + s for s in strengths]) if strengths else '• Some positive factors'}

        <strong>Areas of Concern:</strong>
        {chr(10).join(['• ' + r for r in risk_factors]) if risk_factors else '• Minor concerns'}

        <strong>Recommendations:</strong>
        • Work with an experienced immigration attorney
        • Strengthen weak areas if possible
        • Ensure detailed documentation proving specialty occupation
        • Consider alternative visa options as backup (L-1, O-1)
        • Have employer prepare comprehensive support letter"""
        else:
            recommendation = f"""🔴 <strong>Application Needs Strengthening</strong>

        Your current profile shows a <strong>lower approval probability ({probability:.0%})</strong>.

        <strong>Risk Factors:</strong>
        {chr(10).join(['• ' + r for r in risk_factors]) if risk_factors else '• Multiple concerns identified'}

        {('<strong>Strengths:</strong>' + chr(10) + chr(10).join(['• ' + s for s in strengths])) if strengths else ''}

        <strong>Recommended Actions:</strong>
        • <strong>Consult immigration attorney immediately</strong>
        • Consider delaying to strengthen profile:
        - Negotiate higher salary if possible
        - Gain more relevant work experience
        - Switch to more clearly defined specialty occupation
        • Explore alternative visa categories (L-1, O-1, E-3, TN)
        • If employer is small/startup, ensure very strong documentation
        • Consider multiple candidate applications"""
        
        # ============================================
        # STEP 6: SAVE PREDICTION
        # ============================================
        
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
            'prediction': prediction,
            'strengths': strengths,
            'risk_factors': risk_factors
        })
    
    def get(self, request):
        return redirect('predictions:predict_form')


@method_decorator(login_required, name='dispatch')
class PredictionDetailView(DetailView):
    """View individual prediction details"""
    model = VisaPrediction
    template_name = 'predictions/details.html'
    context_object_name = 'prediction'
    
    def get_queryset(self):
        return VisaPrediction.objects.filter(user=self.request.user)

@method_decorator(login_required, name='dispatch')
class CanadaCalculatorView(View):
    """Calculate Canada Express Entry Comprehensive Ranking System (CRS) score"""
    
    def get(self, request):
        return render(request, 'predictions/canada_form.html')
    
    def post(self, request):
        # Core/Human capital factors
        age = int(request.POST.get('age', 0))
        education = request.POST.get('education', '')
        language_first = request.POST.get('language_first', 'beginner')
        language_second = request.POST.get('language_second', 'none')
        canadian_work = int(request.POST.get('canadian_work', 0))
        foreign_work = int(request.POST.get('foreign_work', 0))
        
        # Additional points
        arranged_employment = request.POST.get('arranged_employment', 'no')
        provincial_nomination = request.POST.get('provincial_nomination', 'no')
        canadian_education = request.POST.get('canadian_education', 'no')
        sibling = request.POST.get('sibling', 'no')
        french_proficiency = request.POST.get('french_proficiency', 'no')
        
        # Calculate points
        total_score = 0
        breakdown = {}
        
        # Age points (max 110 for single applicant)
        age_points_map = {
            18: 90, 19: 90,
            20: 95, 21: 95, 22: 95, 23: 95, 24: 95,
            25: 100, 26: 100, 27: 100, 28: 100, 29: 100,
            30: 105, 31: 99, 32: 94, 33: 88, 34: 83,
            35: 77, 36: 72, 37: 66, 38: 61, 39: 55,
            40: 50, 41: 39, 42: 28, 43: 17, 44: 6,
        }
        
        age_points = age_points_map.get(age, 0) if 18 <= age <= 44 else 0
        total_score += age_points
        breakdown['Age'] = age_points
        
        # Education points (max 150)
        education_points_map = {
            'high_school': 30,
            'one_year': 90,
            'two_year': 98,
            'bachelors': 120,
            'masters': 135,
            'phd': 150
        }
        edu_points = education_points_map.get(education, 0)
        total_score += edu_points
        breakdown['Education'] = edu_points
        
        # First official language (max 136)
        language_points_map = {
            'beginner': 0,      # CLB 4 or less
            'moderate': 32,     # CLB 5-6
            'good': 68,         # CLB 7
            'advanced': 124,    # CLB 8
            'fluent': 136       # CLB 9+
        }
        lang_points = language_points_map.get(language_first, 0)
        total_score += lang_points
        breakdown['First Official Language'] = lang_points
        
        # Second official language (max 24)
        if language_second != 'none':
            second_lang_map = {
                'moderate': 6,   # CLB 5-6
                'good': 12,      # CLB 7
                'advanced': 18,  # CLB 8
                'fluent': 24     # CLB 9+
            }
            second_lang_points = second_lang_map.get(language_second, 0)
            total_score += second_lang_points
            breakdown['Second Official Language'] = second_lang_points
        
        # Canadian work experience (max 80)
        canadian_work_map = {
            0: 0,
            1: 40,
            2: 53,
            3: 64,
            4: 72,
            5: 80
        }
        can_work_points = canadian_work_map.get(min(canadian_work, 5), 80 if canadian_work > 5 else 0)
        total_score += can_work_points
        breakdown['Canadian Work Experience'] = can_work_points
        
        # Skill transferability (max 100)
        # Simplified calculation - education + foreign work combo
        transfer_points = 0
        if foreign_work >= 3 and edu_points >= 120:
            # Good education + foreign experience
            if foreign_work >= 3:
                transfer_points = 25
            if foreign_work >= 4:
                transfer_points = 50
        
        if transfer_points > 0:
            total_score += transfer_points
            breakdown['Skill Transferability'] = transfer_points
        
        # Additional points (max 600)
        additional = 0
        additional_details = []
        
        if arranged_employment == 'yes':
            additional += 200
            additional_details.append('Valid Job Offer: +200')
        
        if provincial_nomination == 'yes':
            additional += 600
            additional_details.append('Provincial Nomination: +600')
        
        if canadian_education == 'yes':
            additional += 30
            additional_details.append('Canadian Education: +30')
        
        if sibling == 'yes':
            additional += 15
            additional_details.append('Sibling in Canada: +15')
        
        if french_proficiency == 'yes':
            additional += 50
            additional_details.append('French Proficiency: +50')
        
        if additional > 0:
            total_score += additional
            breakdown['Additional Points'] = additional
        
        # Determine eligibility and message
        if provincial_nomination == 'yes':
            result_type = 'success'
            message = "🎉 Provincial Nomination guarantees Invitation to Apply!"
            recommendation = "With a Provincial Nomination (+600 points), you are virtually guaranteed to receive an ITA. Proceed with your application immediately."
        elif total_score >= 500:
            result_type = 'success'
            message = f"🟢 Excellent Score! Very likely to receive ITA"
            recommendation = "Your score is well above recent cutoffs (typically 470-500). Monitor Express Entry draws and prepare your documentation."
        elif total_score >= 470:
            result_type = 'warning'
            message = f"🟡 Good Score - Monitor draw cutoffs closely"
            recommendation = "Your score is competitive but near the cutoff line. Consider: improving language scores (IELTS/TEF), gaining more work experience, or pursuing provincial nomination."
        elif total_score >= 440:
            result_type = 'warning'
            message = f"🟡 Moderate Score - Consider improvements"
            recommendation = "Your score is below recent cutoffs. Focus on: achieving higher language scores (aim for CLB 9+), upgrading education, or securing a provincial nomination."
        else:
            result_type = 'danger'
            message = f"🔴 Score Below Cutoffs - Improvement Needed"
            recommendation = "Your current score is significantly below recent cutoffs. Priority actions: improve language proficiency to CLB 9+, consider additional education, gain more work experience, or explore provincial nominee programs."
        
        context = {
            'total_score': total_score,
            'breakdown': breakdown,
            'result_type': result_type,
            'message': message,
            'recommendation': recommendation,
            'additional_details': additional_details,
            'form_data': request.POST,  # To repopulate form
        }
        
        return render(request, 'predictions/canada_result.html', context)
