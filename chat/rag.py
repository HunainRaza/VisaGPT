"""
RAG (Retrieval Augmented Generation) System
Uses OpenAI embeddings and MongoDB for vector search
"""
import os
import openai
from openai import OpenAI
from pymongo import MongoClient
import numpy as np
from typing import List, Dict

def load_config():
    """Load configuration from Django settings or environment"""
    config = {
        'OPENAI_API_KEY': None,
        'MONGODB_URI': 'mongodb://localhost:27017/'
    }
    
    # Try Django settings first
    try:
        from django.conf import settings
        if settings.configured:
            config['OPENAI_API_KEY'] = getattr(settings, 'OPENAI_API_KEY', None)
            config['MONGODB_URI'] = getattr(settings, 'MONGODB_URI', 'mongodb://localhost:27017/')
            print("✓ Loaded config from Django settings")
            return config
    except (ImportError, AttributeError):
        pass
    
    # Fall back to loading dev.env manually
    try:
        import environ
        env = environ.Env()
        if os.path.exists('dev.env'):
            environ.Env.read_env('dev.env')
            print("✓ Loaded dev.env")
    except ImportError:
        print("⚠️ django-environ not installed, trying python-dotenv")
        try:
            from dotenv import load_dotenv
            if os.path.exists('dev.env'):
                load_dotenv('dev.env')
                print("✓ Loaded dev.env with python-dotenv")
        except ImportError:
            print("⚠️ No .env loader found")
    
    # Get from environment
    config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    config['MONGODB_URI'] = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    
    return config


# Load configuration at module level
_config = load_config()

# Set global API key
openai.api_key = _config['OPENAI_API_KEY']

class VisaRAG:
    def __init__(self):
        # Verify API key is set
        if not openai.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found! "
                "Please set it in Django settings or dev.env file."
            )
        
        # Create OpenAI client
        self.client = OpenAI()
        
        # MongoDB setup
        mongo_uri = _config['MONGODB_URI']
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client['visagpt']
        self.collection = self.db['Visa-collection']
        
        print(f"✓ VisaRAG initialized")
        print(f"  - OpenAI: Connected")
        print(f"  - MongoDB: {mongo_uri}")
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def add_knowledge(self, country: str, visa_type: str, content: str, source_url: str = ""):
        """Add visa knowledge to database with embeddings"""
        embedding = self.generate_embedding(content)
        
        doc = {
            'country': country,
            'visa_type': visa_type,
            'content': content,
            'embedding': embedding,
            'source_url': source_url
        }
        
        self.collection.insert_one(doc)
        print(f"✓ Added: {country} - {visa_type}")
    
    def search_knowledge(self, query: str, country: str = None, top_k: int = 3) -> List[Dict]:
        """Search for relevant visa information using cosine similarity"""
        query_embedding = self.generate_embedding(query)
        
        filter_dict = {}
        if country:
            filter_dict['country'] = country
        
        results = []
        for doc in self.collection.find(filter_dict):
            similarity = np.dot(query_embedding, doc['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc['embedding'])
            )
            results.append({
                'content': doc['content'],
                'country': doc['country'],
                'visa_type': doc['visa_type'],
                'source_url': doc.get('source_url', ''),
                'similarity': similarity
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def generate_response(self, user_query: str, conversation_history: List[Dict] = None) -> str:
        """Generate AI response with RAG context"""
        try:
            context_docs = self.search_knowledge(user_query)
            
            context = "\n\n".join([
                f"**{doc['country']} - {doc['visa_type']}**\n{doc['content']}"
                for doc in context_docs
            ])
            
            messages = [
                {"role": "system", "content": f"""You are VisaGPT, an expert immigration assistant.

Use the following visa information to provide accurate guidance:

{context}

Provide clear, helpful advice. If information isn't in the context, say so and suggest the user consult official sources."""}
            ]
            
            if conversation_history:
                messages.extend(conversation_history[-5:])
            
            messages.append({"role": "user", "content": user_query})
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            raise  # Re-raise to be caught by the view
    
    def initialize_knowledge_base(self):
        """Initialize with visa knowledge for major countries"""
        # Check if already initialized
        existing_count = self.collection.count_documents({})
        if existing_count > 0:
            print(f"\n⚠️ Knowledge base already has {existing_count} documents.")
            print("Skipping initialization.")
            return
        
        visa_knowledge = [
            {
                'country': 'USA',
                'visa_type': 'H-1B',
                'content': """H-1B Work Visa Requirements:
                - Bachelor's degree or equivalent in specialty occupation
                - Job offer from US employer (must be specialty occupation)
                - Employer must file petition and pay fees
                - Prevailing wage requirements must be met
                - Annual cap: 85,000 (65k regular + 20k advanced degree)
                - Application period: April 1st each year
                - Valid for 3 years, extendable to 6 years
                - Processing: 3-6 months (premium processing available for extra fee)
                - Spouse and children can get H-4 dependent visas""",
                'source_url': 'https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations'
            },
            {
                'country': 'USA',
                'visa_type': 'F-1',
                'content': """F-1 Student Visa Requirements:
                - Acceptance at SEVP-approved US school
                - Proof of financial support for tuition and living expenses
                - Intent to return to home country after studies
                - English proficiency (TOEFL/IELTS scores)
                - Valid passport (must be valid for at least 6 months)
                - I-20 form from school
                - SEVIS fee payment ($350)
                - Can work on-campus up to 20 hours/week during semester
                - OPT (Optional Practical Training) available after graduation:
                * 12 months for all majors
                * 24 months extension for STEM majors
                - CPT (Curricular Practical Training) available during studies""",
                'source_url': 'https://www.uscis.gov/f-1-students'
            },
            {
                'country': 'Canada',
                'visa_type': 'Express Entry',
                'content': """Canada Express Entry Requirements:
                - Points-based system (Comprehensive Ranking System - CRS)
                - Minimum 67 points out of 100 on FSW points grid
                - Key factors: Age (under 30 gets max points), Education (ECA required), Work Experience, Language (CLB 7+)
                - Language tests: IELTS/CELPIP for English, TEF for French
                - Educational Credential Assessment (ECA) required for foreign degrees
                - Proof of funds: $13,310 CAD for single person (more for family)
                - Processing time: 6 months average
                - Three programs: Federal Skilled Worker, Federal Skilled Trades, Canadian Experience Class
                - Create Express Entry profile and wait for Invitation to Apply (ITA)
                - CRS score cutoff varies (typically 470-500+)""",
                'source_url': 'https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/express-entry.html'
            },
            {
                'country': 'UK',
                'visa_type': 'Skilled Worker',
                'content': """UK Skilled Worker Visa Requirements:
                - Job offer from UK employer with valid sponsor license
                - Certificate of Sponsorship (CoS) from employer
                - Minimum salary: £26,200 or going rate for job (whichever is higher)
                - English language requirement: B1 level on CEFR scale
                - Job must be on eligible occupations list (RQF Level 3+)
                - Healthcare surcharge: £624 per year
                - Valid for up to 5 years
                - Can bring family members (dependents)
                - Can lead to settlement (Indefinite Leave to Remain) after 5 years
                - Switching from Student visa possible in UK""",
                'source_url': 'https://www.gov.uk/skilled-worker-visa'
            },
            {
                'country': 'Australia',
                'visa_type': 'Skilled Independent',
                'content': """Australia Skilled Independent Visa (189) Requirements:
                - Points-based system (minimum 65 points required)
                - Occupation must be on skilled occupation list
                - Skills assessment for your occupation
                - Age: Under 45 years old
                - English language: Competent English (IELTS 6 each band minimum)
                - Expression of Interest (EOI) through SkillSelect
                - Points: Age (max 30), English (max 20), Experience (max 20), Education (max 20)
                - Permanent residence visa
                - No sponsor required
                - Processing time: 8-12 months
                - Health and character requirements""",
                'source_url': 'https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189'
            }
        ]
        
        print("\nAdding knowledge to database...")
        for knowledge in visa_knowledge:
            self.add_knowledge(**knowledge)
        
        print(f"\n✓ Initialized knowledge base with {len(visa_knowledge)} entries")


if __name__ == '__main__':
    print("=" * 60)
    print("INITIALIZING RAG SYSTEM")
    print("=" * 60)
    print()
    
    try:
        rag = VisaRAG()
        
        print("\nPopulating knowledge base...")
        rag.initialize_knowledge_base()
        
        print("\nTesting search...")
        results = rag.search_knowledge("H-1B visa requirements", country="USA")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['visa_type']} (similarity: {result['similarity']:.3f})")
            print(result['content'][:200] + "...")
        
        print("\n" + "=" * 60)
        print("✓ RAG system ready!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()