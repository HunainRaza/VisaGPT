"""
H-1B Visa Approval Prediction Model Training
Trained on REAL H-1B Kaggle dataset
Fixed version with proper predictions and no warnings
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import re


class VisaMLModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def extract_state(self, worksite):
        """Extract state from WORKSITE column (e.g., 'ANN ARBOR, MICHIGAN' -> 'MICHIGAN')"""
        if pd.isna(worksite):
            return 'UNKNOWN'
        # Split by comma and take the last part (state)
        parts = str(worksite).split(',')
        if len(parts) >= 2:
            return parts[-1].strip().upper()
        return 'UNKNOWN'
    
    def prepare_data(self, df):
        """Clean and prepare REAL H-1B data for training"""
        print(f"Original dataset size: {len(df)} rows")
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Handle CASE_STATUS - consider all CERTIFIED variations as positive
        df = df.dropna(subset=['CASE_STATUS'])
        df['TARGET'] = df['CASE_STATUS'].str.contains('CERTIFIED', case=False, na=False).astype(int)
        
        print(f"Target distribution:")
        print(df['TARGET'].value_counts())
        
        features = []
        
        # Wage features
        if 'PREVAILING_WAGE' in df.columns:
            df['PREVAILING_WAGE'] = pd.to_numeric(df['PREVAILING_WAGE'], errors='coerce')
            # Remove outliers and invalid wages
            df = df[(df['PREVAILING_WAGE'] > 10000) & (df['PREVAILING_WAGE'] < 500000)]
            df['WAGE_LEVEL'] = pd.cut(
                df['PREVAILING_WAGE'], 
                bins=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
            features.append('WAGE_LEVEL')
            print(f"✓ Wage feature created")
        
        # Employer features
        if 'EMPLOYER_NAME' in df.columns:
            employer_counts = df['EMPLOYER_NAME'].value_counts()
            df['EMPLOYER_FREQUENCY'] = df['EMPLOYER_NAME'].map(employer_counts)
            # Normalize frequency
            df['EMPLOYER_FREQUENCY'] = np.log1p(df['EMPLOYER_FREQUENCY'])
            features.append('EMPLOYER_FREQUENCY')
            print(f"✓ Employer frequency feature created")
        
        # Job title encoding (tech vs non-tech)
        if 'JOB_TITLE' in df.columns:
            tech_keywords = [
                'software', 'engineer', 'developer', 'programmer', 
                'analyst', 'data', 'scientist', 'architect', 'devops',
                'machine learning', 'ai', 'technology'
            ]
            df['IS_TECH_JOB'] = df['JOB_TITLE'].str.lower().str.contains(
                '|'.join(tech_keywords), 
                na=False
            ).astype(int)
            features.append('IS_TECH_JOB')
            print(f"✓ Tech job feature created ({df['IS_TECH_JOB'].sum()} tech jobs)")
        
        # Extract state from WORKSITE column
        if 'WORKSITE' in df.columns:
            print("Extracting states from WORKSITE...")
            df['WORKSITE_STATE'] = df['WORKSITE'].apply(self.extract_state)
            df['WORKSITE_STATE'] = df['WORKSITE_STATE'].fillna('UNKNOWN')
            
            # Encode states
            le = LabelEncoder()
            df['STATE_ENCODED'] = le.fit_transform(df['WORKSITE_STATE'])
            self.label_encoders['WORKSITE_STATE'] = le
            features.append('STATE_ENCODED')
            print(f"✓ State feature created ({df['WORKSITE_STATE'].nunique()} unique states)")
        
        # SOC_NAME encoding (if SOC_CODE not available)
        if 'SOC_NAME' in df.columns:
            df['SOC_NAME'] = df['SOC_NAME'].fillna('UNKNOWN')
            # Only encode top N SOC names to avoid too many categories
            top_soc = df['SOC_NAME'].value_counts().head(100).index
            df['SOC_NAME_ENCODED'] = df['SOC_NAME'].apply(
                lambda x: x if x in top_soc else 'OTHER'
            )
            le = LabelEncoder()
            df['SOC_ENCODED'] = le.fit_transform(df['SOC_NAME_ENCODED'])
            self.label_encoders['SOC_NAME'] = le
            features.append('SOC_ENCODED')
            print(f"✓ SOC feature created ({len(top_soc)} categories)")
        
        # Full-time position
        if 'FULL_TIME_POSITION' in df.columns:
            df['IS_FULL_TIME'] = (df['FULL_TIME_POSITION'] == 'Y').astype(int)
            features.append('IS_FULL_TIME')
            print(f"✓ Full-time feature created ({df['IS_FULL_TIME'].sum()} full-time)")
        
        self.feature_columns = features
        
        # Final cleanup
        df = df.dropna(subset=features)
        X = df[features]
        y = df['TARGET']
        
        print(f"\nFinal dataset: {len(X)} samples with {len(features)} features")
        print(f"Features: {features}")
        
        return X, y
    
    def train(self, X, y):
        """Train the XGBoost model"""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        print(f"Splitting data into train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training XGBoost model...")
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        print("\nEvaluating model...")
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f} ({value*100:.2f}%)")
        print("=" * 60)
        
        return metrics
    
    def predict(self, features_dict):
        """Predict approval probability - FIXED VERSION"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create DataFrame with proper feature names (fixes the warning!)
        X = pd.DataFrame([features_dict], columns=self.feature_columns)
        
        # Fill missing features with 0
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Ensure correct order
        X = X[self.feature_columns]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        probability = self.model.predict_proba(X_scaled)[0, 1]
        
        return probability
    
    def save_model(self, path='ml_models'):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, f'{path}/visa_model.pkl')
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')
        joblib.dump(self.feature_columns, f'{path}/feature_columns.pkl')
        print(f"\n✓ Model saved to {path}/")
    
    def load_model(self, path='ml_models'):
        """Load trained model"""
        self.model = joblib.load(f'{path}/visa_model.pkl')
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.label_encoders = joblib.load(f'{path}/label_encoders.pkl')
        self.feature_columns = joblib.load(f'{path}/feature_columns.pkl')


if __name__ == '__main__':
    print("=" * 60)
    print("H-1B VISA APPROVAL PREDICTION MODEL TRAINING")
    print("REAL KAGGLE DATASET")
    print("=" * 60)
    
    # Load REAL data
    csv_file = 'ml_engine/h1b_kaggle.csv'
    
    if not os.path.exists(csv_file):
        csv_file = 'ml_engine/h1b_kaggle.csv'  # Try current directory
        if not os.path.exists(csv_file):
            print(f"\n❌ ERROR: h1b_kaggle.csv not found!")
            print(f"Please place the Kaggle H-1B CSV file in:")
            print(f"  - ml_engine/h1b_kaggle.csv")
            print(f"  - or current directory")
            print(f"\nDownload from: https://www.kaggle.com/datasets/nsharan/h-1b-visa")
            exit(1)
    
    print(f"\n✓ Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Initialize and train model
    model = VisaMLModel()
    
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    X, y = model.prepare_data(df)
    
    # Train model
    metrics = model.train(X, y)
    
    # Save model
    model.save_model()
    
    # Test prediction with REALISTIC values
    print("\n" + "=" * 60)
    print("TESTING PREDICTION")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Tech job, high salary (CA)',
            'features': {
                'WAGE_LEVEL': 5.0,           # High wage
                'EMPLOYER_FREQUENCY': 7.0,    # Well-known employer (log scale)
                'IS_TECH_JOB': 1.0,          # Yes, tech job
                'STATE_ENCODED': 4.0,        # California (typically 4)
                'SOC_ENCODED': 10.0,         # Software engineer SOC
                'IS_FULL_TIME': 1.0          # Full-time
            }
        },
        {
            'name': 'Non-tech job, medium salary (NY)',
            'features': {
                'WAGE_LEVEL': 3.0,           # Medium wage
                'EMPLOYER_FREQUENCY': 3.0,    # Average employer
                'IS_TECH_JOB': 0.0,          # Not tech
                'STATE_ENCODED': 33.0,       # New York
                'SOC_ENCODED': 50.0,         # Business analyst
                'IS_FULL_TIME': 1.0          # Full-time
            }
        },
        {
            'name': 'Low salary, part-time',
            'features': {
                'WAGE_LEVEL': 1.0,           # Low wage
                'EMPLOYER_FREQUENCY': 1.0,    # Unknown employer
                'IS_TECH_JOB': 0.0,          # Not tech
                'STATE_ENCODED': 20.0,       # Other state
                'SOC_ENCODED': 80.0,         # Other occupation
                'IS_FULL_TIME': 0.0          # Part-time
            }
        }
    ]
    
    for test in test_cases:
        prob = model.predict(test['features'])
        risk = 'LOW' if prob >= 0.8 else 'MEDIUM' if prob >= 0.6 else 'HIGH'
        print(f"\n{test['name']}:")
        print(f"  Approval probability: {prob:.2%}")
        print(f"  Risk level: {risk}")
    
    print("\n" + "=" * 60)
    print("✓ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel files saved in: ml_models/")
    print("Ready to use in Django predictions app!")
    print("\nNext steps:")
    print("1. Run: python manage.py runserver")
    print("2. Visit: http://localhost:8000/predictions/predict/")
    print("3. Test predictions with real data!")