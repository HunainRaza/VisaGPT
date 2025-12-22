import os
import environ
from core.settings.common import *

# Initialize environment variables
env = environ.Env(
    # Set default values with production-appropriate settings
    DEBUG=(bool, False),
    SECRET_KEY=(str, ''),
    ALLOWED_HOSTS=(list, []),
    DB_NAME=(str, ''),
    DB_USER=(str, ''),
    DB_PASSWORD=(str, ''),
    DB_HOST=(str, ''),
    DB_PORT=(str, '5432'),
)

# Read .env file if it exists (optional for production)
environ.Env.read_env(os.path.join(BASE_DIR, 'prod.env'))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY')

# SECURITY WARNING: set debug to False in production!
DEBUG = env('DEBUG')

# Configure allowed hosts
ALLOWED_HOSTS = env('ALLOWED_HOSTS')

# Database Configuration - Supabase PostgreSQL
# DATABASES = {
#     'default': {
#         "ENGINE": "django.db.backends.postgresql_psycopg2",
#         "NAME": env('SUPABASE_DB_NAME'),
#         "USER": env('SUPABASE_DB_USER'),
#         "PASSWORD": env('SUPABASE_DB_PASSWORD'),
#         "HOST": env('SUPABASE_DB_HOST'),
#         "PORT": env('SUPABASE_DB_PORT', default='5432'),
#         'ATOMIC_REQUESTS': True,
#         'OPTIONS': {
#             'sslmode': 'require',  # Supabase requires SSL connections
#         }
#     }
# }

# Security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Cors and Security
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = env.list('CORS_ALLOWED_ORIGINS', default=[])

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
}

# Email settings (override from common.py)
EMAIL_HOST_PASSWORD = env('SENDGRID_API_KEY')
DEFAULT_FROM_EMAIL = env('DEFAULT_FROM_EMAIL')

# Add OpenAI API key to settings
OPENAI_API_KEY = env('OPENAI_API_KEY')
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')

# Static and Media Files for Production
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'

# Installed Apps (add any production-specific apps)
INSTALLED_APPS += [
    'whitenoise.runserver_nostatic',
    'corsheaders',
]

# Middleware (add CORS middleware)
MIDDLEWARE.insert(0, 'corsheaders.middleware.CorsMiddleware')
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')

# Supabase Storage Configuration
SUPABASE_URL = env('SUPABASE_URL')
SUPABASE_ACCESS_KEY = env('SUPABASE_ACCESS_KEY')  # The anon/public key
SUPABASE_SECRET_KEY = env('SUPABASE_SECRET_KEY')  # The service_role key (for more secure operations)
SUPABASE_STORAGE_BUCKET_NAME = env('SUPABASE_STORAGE_BUCKET_NAME', default='my-portfolio')

# Media configuration
DEFAULT_FILE_STORAGE = 'core.storage_backends.SupabaseStorage'
MEDIA_URL = '/media/'  # This will be prefixed to your URLs in templates

# Allow Supabase Storage domain in CORS settings if needed
if SUPABASE_URL:
    parsed_url = SUPABASE_URL.rstrip('/').split('/')[-1]
    supabase_domain = f"https://{parsed_url}.supabase.co"
    if supabase_domain not in CORS_ALLOWED_ORIGINS:
        CORS_ALLOWED_ORIGINS.append(supabase_domain)
