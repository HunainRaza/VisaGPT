# 🌍 VisaGPT - AI-Powered Immigration Assistant

> **Built for VisaVerse AI Hackathon 2025**  
> AI-powered immigration assistant combining ML, RAG, and intelligent document analysis

[![Django](https://img.shields.io/badge/Django-4.2-green.svg)](https://www.djangoproject.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Vector%20Search-green.svg)](https://www.mongodb.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-93.95%25%20Accuracy-red.svg)](https://xgboost.readthedocs.io/)

---

## 🎯 Project Overview

**VisaGPT** is an intelligent immigration assistant that combines **Machine Learning**, **Retrieval-Augmented Generation (RAG)**, and **Document Analysis** to help users navigate complex visa application processes with confidence.

### 🏆 Key Achievements

- ✅ **93.95% ML Accuracy** on 3+ million H-1B visa records
- ✅ **Real RAG System** with GPT-4 and MongoDB vector search
- ✅ **Production-Ready** full-stack Django application
- ✅ **Professional UI** with ChatGPT-style interface
- ✅ **Complete CRUD** for document management

---

## ✨ Features

### 🤖 1. AI-Powered Chatbot with RAG
- **GPT-4 Integration** for natural language processing
- **MongoDB Vector Search** for semantic document retrieval
- **Context-Aware Responses** using conversation history
- **Real-Time Chat** with typing indicators
- **Knowledge Base** covering USA, Canada, UK, and Australia visa requirements

### 📊 2. ML-Powered Visa Predictions
- **Real XGBoost Model** trained on 3 million H-1B visa applications from Kaggle
- **Genuine Machine Learning** - not fake or rule-based
- **Risk Assessment** with personalized recommendations
- **Interactive Dashboard** with Chart.js visualizations
- **Approval Probability** calculations based on applicant profile

### 📄 3. Document Management System
- **Secure Upload** for passport, diplomas, bank statements, etc.
- **File Organization** by document type
- **Document Preview** and download functionality
- **Delete Confirmation** modals for safety
- **OCR Ready** - prepared for future text extraction

### 📈 4. Professional Dashboard
- **Real-time Statistics** on predictions and documents
- **Chart.js Visualizations** (bar charts, pie charts)
- **Recent Predictions** table with color-coded risk levels
- **Quick Actions** for common tasks
- **Responsive Design** for all devices

---

## 🛠️ Tech Stack

### **Backend**
- **Django 4.2** - Full-stack web framework
- **Python 3.10+** - Core programming language
- **PostgreSQL** - Primary database for relational data
- **MongoDB** - Vector database for RAG embeddings

### **AI/ML**
- **OpenAI GPT-4** - Conversational AI
- **XGBoost** - Gradient boosting for predictions
- **Scikit-learn** - Feature engineering and preprocessing
- **OpenAI Embeddings** - Vector representations for RAG

### **Frontend**
- **Bootstrap 5.3** - Responsive UI framework
- **Chart.js** - Data visualizations
- **Animate.css** - Smooth animations
- **Bootstrap Icons** - Icon library
- **Custom CSS** - Professional styling

### **Data**
- **Kaggle H-1B Dataset** - 3+ million visa records (2011-2016)
- **MongoDB Atlas** - Cloud vector database
- **PostgreSQL** - User data and predictions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                    │
│  (Bootstrap 5 + Custom CSS + Chart.js + HTMX)       │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                Django Backend                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Accounts │  │   Chat   │  │Documents │           │
│  │   App    │  │   App    │  │   App    │           │
│  └──────────┘  └──────────┘  └──────────┘           │
│  ┌─────────────┐  ┌──────────┐                      │
│  │Predictions  │  │ML Engine │                      │
│  │   App       │  │   App    │                      │
│  └─────────────┘  └──────────┘                      │
└───────┬────────────┬────────────┬───────────────────┘
        │            │            │
┌───────▼──────┐ ┌──▼──────┐ ┌──▼────────┐
│  PostgreSQL  │ │ MongoDB │ │  OpenAI   │
│   Database   │ │  Atlas  │ │  API      │
│              │ │ (Vector │ │ (GPT-4 +  │
│ - Users      │ │ Search) │ │Embeddings)│
│ - Predictions│ │         │ │           │
│ - Documents  │ │ - RAG   │ │ - Chat    │
│ - Chat Logs  │ │ - Docs  │ │ - Vectors │
└──────────────┘ └─────────┘ └───────────┘
```

### **Data Flow: Chat with RAG**
```
User Query → Embedding → Vector Search → Relevant Docs
     ↓                                         ↓
GPT-4 ← Context (Query + Docs + History) ←────┘
     ↓
AI Response → Save to Database → Display to User
```

### **Data Flow: ML Prediction**
```
User Input → Feature Engineering → XGBoost Model
     ↓              ↓                    ↓
  (Job, Salary, Education)    (6 features)  (Probability)
     ↓
Risk Assessment → Recommendations → Save → Display
```

---

## 📦 Installation & Setup

### **Prerequisites**
- Python 3.10+
- PostgreSQL 14+
- MongoDB Atlas account (free tier)
- OpenAI API key
- Git

### **1. Clone Repository**
```bash
git clone https://github.com/HunainRaza/VisaGPT.git
cd VisaGPT
```

### **2. Create Virtual Environment**
```bash
python -m venv env

# Windows
env\Scripts\activate

# Mac/Linux
source env/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Environment Variables**
Create a `.env` file in the root directory:

```env
# Django
SECRET_KEY=your-secret-key-here
DEBUG=True

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/visagpt

# MongoDB
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DB_NAME=visagpt

# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional
ALLOWED_HOSTS=localhost,127.0.0.1
```

### **5. Database Setup**
```bash
# Create PostgreSQL database
createdb visagpt

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### **6. Download Dataset (Optional - for training)**
See [ml_engine/DATA_README.md](ml_engine/DATA_README.md) for instructions on downloading the H-1B dataset if you want to retrain the model.

**Note:** Pre-trained models are already included in `ml_models/`

### **7. Initialize RAG System**
```bash
# This populates MongoDB with visa knowledge base
python manage.py shell
>>> from chat.rag import VisaRAG
>>> rag = VisaRAG()
>>> # Knowledge base is now initialized!
```

### **8. Run Development Server**
```bash
python manage.py runserver
```

Visit: `http://localhost:8000`

---

## 🚀 Deployment

### **Production Checklist**
- [ ] Set `DEBUG=False` in environment
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Set up PostgreSQL production database
- [ ] Configure MongoDB Atlas
- [ ] Set secure `SECRET_KEY`
- [ ] Configure static files serving
- [ ] Run `python manage.py collectstatic`
- [ ] Set up SSL certificate

### **Deployment Platforms**
- **Recommended:** Render, Railway, or Heroku
- **Database:** PostgreSQL (included with most platforms)
- **File Storage:** AWS S3 or Cloudinary for documents

---

## 📊 ML Model Details

### **Dataset**
- **Source:** Kaggle H-1B Visa Petitions (2011-2016)
- **Records:** 3,002,458 applications
- **Size:** 469 MB (not included in repo - see DATA_README.md)
- **Download:** https://www.kaggle.com/datasets/nsharan/h-1b-visa

### **Model Performance**
```
Algorithm: XGBoost Classifier
Accuracy: 93.95%
Precision: 94.2%
Recall: 93.7%
F1-Score: 93.9%
Training Samples: 2,401,966
Testing Samples: 600,492
```

### **Features Used**
1. **IS_TECH_JOB** - Binary flag for tech occupations
2. **WAGE_LEVEL** - Salary bracket (1-5)
3. **IS_FULL_TIME** - Employment type
4. **EMPLOYER_FREQUENCY** - Employer's historical applications
5. **STATE_ENCODED** - Location encoding
6. **SOC_ENCODED** - Occupation code encoding

### **Training Command**
```bash
# Download dataset first (see DATA_README.md)
python ml_engine/train_model.py
```

---

## 🎨 Screenshots

### Landing Page
![Landing Page](screenshots/landing-page.png)

### Chat Interface
![Chat with RAG](screenshots/Chat-interface.png)

### ML Predictions Dashboard
![Dashboard](screenshots/ML-Predictions-Dashboard.png)

### Document Management
![Documents](screenshots/Document-Management.png)

---

## 🧪 Testing

```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test accounts
python manage.py test chat
python manage.py test predictions
python manage.py test documents
```

---

## 📂 Project Structure

```
VisaGPT/
├── accounts/                 # User authentication & profiles
│   ├── models.py
│   ├── views.py            # CBVs for register/login/profile
│   ├── forms.py            # User creation forms
│   └── templates/
├── chat/                     # AI chatbot with RAG
│   ├── models.py           # Conversation & message models
│   ├── views.py            # Chat interface & API
│   ├── rag.py              # RAG implementation
│   └── templates/
├── predictions/              # ML visa predictions
│   ├── models.py           # Prediction results
│   ├── views.py            # Prediction logic
│   └── templates/
├── documents/                # Document management
│   ├── models.py           # Document storage
│   ├── views.py            # Upload/list/detail/delete
│   └── templates/
├── ml_engine/                # ML training pipeline
│   ├── train_model.py      # Model training script
│   ├── DATA_README.md      # Dataset instructions
│   └── h1b_kaggle.csv      # (Download separately)
├── ml_models/                # Trained models
│   ├── visa_model.pkl      # XGBoost classifier
│   ├── label_encoders.pkl  # Encoders
│   ├── scaler.pkl          # Normalizer
│   └── feature_columns.pkl # Features
├── static/                   # CSS, JS, images
│   └── css/
│       └── style.css       # Custom styling
├── templates/                # Global templates
│   ├── base.html
│   └── ...
├── core/                     # Django settings
│   ├── settings/
│   │   ├── common.py       # Shared settings
│   │   ├── development.py  # Dev settings
│   │   └── production.py   # Prod settings
│   └── urls.py
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔑 Key Features Explained

### **1. Real Machine Learning**
Unlike many hackathon projects that fake ML with rules, VisaGPT uses a **genuine XGBoost model** trained on 3 million real visa applications. The model achieves 93.95% accuracy through proper feature engineering and hyperparameter tuning.

### **2. Authentic RAG System**
The chatbot doesn't just call GPT-4 - it implements **true Retrieval-Augmented Generation**:
- User query → Embedded via OpenAI
- Vector search in MongoDB for relevant visa info
- Retrieved context + query → GPT-4
- Grounded, accurate responses with source data

### **3. Production-Ready Code**
- **Class-Based Views** (not function views)
- **Proper separation of concerns**
- **Environment-based settings**
- **Database migrations excluded** from repo
- **Security best practices**

---

## 🎯 Hackathon Alignment

### **VisaVerse AI Hackathon Themes**

✅ **Global Mobility** - Removes barriers to international movement
✅ **AI-Driven Solutions** - Genuine ML and RAG implementation
✅ **Document Processing** - Intelligent document management
✅ **User Experience** - Professional, intuitive interface

---

## 🤝 Contributing

This project was built for the VisaVerse AI Hackathon 2025. After the hackathon:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👤 Author

**Hunain Raza**
- GitHub: [@HunainRaza](https://github.com/HunainRaza)
- Email: hunainrazazaidi@gmail.com

---

## 🙏 Acknowledgments

- **VisaVerse AI Hackathon** for organizing this amazing competition
- **Kaggle** for the H-1B visa dataset
- **OpenAI** for GPT-4 and embeddings API
- **MongoDB** for vector search capabilities
- **Django Community** for the excellent framework

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact via email: hunainrazazaidi@gmail.com

---

## ⭐ Star This Repo!

If this project helped you or you found it interesting, please give it a star! ⭐

---

**Built with ❤️ for VisaVerse AI Hackathon 2025**

*Empowering global mobility through artificial intelligence*