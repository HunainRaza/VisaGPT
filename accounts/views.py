from django.shortcuts import redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views.generic import CreateView, FormView, View, TemplateView
from django.urls import reverse_lazy
from .forms import SimpleUserCreationForm


class RegisterView(CreateView):
    """
    User registration with class-based view
    
    Choose your form:
    - SimpleUserCreationForm: Easy registration (recommended for hackathon)
    """
    form_class = SimpleUserCreationForm
    
    template_name = 'accounts/register.html'
    success_url = reverse_lazy('chat:chat')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Create Account'
        return context
    
    def form_valid(self, form):
        """Called when form is valid - create user and log them in"""
        user = form.save()
        login(self.request, user)
        messages.success(
            self.request, 
            f'Welcome {user.username}! Your account has been created successfully.'
        )
        return super().form_valid(form)
    
    def form_invalid(self, form):
        """Called when form has errors"""
        messages.error(
            self.request,
            'Please correct the errors below.'
        )
        return super().form_invalid(form)


class LoginView(FormView):
    """User login with class-based view"""
    form_class = AuthenticationForm
    template_name = 'accounts/login.html'
    success_url = reverse_lazy('chat:chat')
    
    def get_form_kwargs(self):
        """Pass request to form"""
        kwargs = super().get_form_kwargs()
        kwargs['request'] = self.request
        return kwargs
    
    def form_valid(self, form):
        """Called when login is successful"""
        user = form.get_user()
        login(self.request, user)
        messages.success(self.request, f'Welcome back, {user.username}!')
        
        # Check if there's a 'next' parameter
        next_url = self.request.GET.get('next')
        if next_url:
            return redirect(next_url)
        return super().form_valid(form)
    
    def form_invalid(self, form):
        """Called when login fails"""
        messages.error(self.request, 'Invalid username or password.')
        return super().form_invalid(form)


class LogoutView(View):
    """User logout with class-based view"""
    def get(self, request):
        logout(request)
        messages.info(request, 'You have been logged out successfully.')
        return redirect('accounts:login')
    
    def post(self, request):
        """Also handle POST for logout"""
        logout(request)
        messages.info(request, 'You have been logged out successfully.')
        return redirect('accounts:login')


@method_decorator(login_required, name='dispatch')
class ProfileView(TemplateView):
    """User profile with class-based view"""
    template_name = 'accounts/profile.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        # Add user's predictions and documents count
        context['predictions_count'] = self.request.user.predictions.count()
        context['documents_count'] = self.request.user.documents.count()
        return context
    
    def post(self, request):
        """Handle profile update"""
        user = request.user
        
        # Update email
        email = request.POST.get('email', '').strip()
        if email:
            user.email = email
        
        # Update names
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        
        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        
        user.save()
        
        messages.success(request, 'Profile updated successfully!')
        return redirect('accounts:profile')