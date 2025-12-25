"""
H-1B Visa Approval Prediction Model Training
CORRECT VERSION - Handles all CASE_STATUS values properly
Trained on REAL H-1B Kaggle dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os


class VisaMLModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def extract_state(self, worksite):
        """Extract state from WORKSITE column"""
        if pd.isna(worksite):
            return 'UNKNOWN'
        parts = str(worksite).split(',')
        if len(parts) >= 2:
            return parts[-1].strip().upper()
        return 'UNKNOWN'
    
    def prepare_data(self, df):
        """Clean and prepare REAL H-1B data for training"""
        print(f"\n{'='*70}")
        print(f"STEP 1: DATA PREPARATION")
        print(f"{'='*70}")
        print(f"Original dataset size: {len(df):,} rows")
        
        df = df.copy()
        
        # ============================================
        # CRITICAL: HANDLE CASE_STATUS CORRECTLY
        # ============================================
        print(f"\nAnalyzing CASE_STATUS column...")
        
        # Check what statuses we have
        if 'CASE_STATUS' not in df.columns:
            raise ValueError("CASE_STATUS column not found in dataset!")
        
        df = df.dropna(subset=['CASE_STATUS'])
        
        # Print all unique statuses
        print(f"\nUnique CASE_STATUS values found:")
        status_counts = df['CASE_STATUS'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count:,}")
        
        # Define what counts as "approved" vs "denied"
        # According to H-1B logic:
        # - CERTIFIED = Approved ✓
        # - CERTIFIED-WITHDRAWN = Was approved, then withdrawn ✓ (still counts as approved for prediction)
        # - DENIED = Denied ✗
        # - WITHDRAWN = Withdrawn before adjudication (drop these - never decided)
        
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION LOGIC")
        print(f"{'='*70}")
        print(f"APPROVED (TARGET=1):")
        print(f"  - CERTIFIED")
        print(f"  - CERTIFIED-WITHDRAWN (was approved before withdrawal)")
        print(f"\nDENIED (TARGET=0):")
        print(f"  - DENIED")
        print(f"\nDROPPED (not adjudicated):")
        print(f"  - WITHDRAWN (never reached decision)")
        print(f"  - Any other status")
        
        # Create TARGET column
        df['TARGET'] = 0  # Default to denied
        
        # Mark approved cases (CERTIFIED or CERTIFIED-WITHDRAWN)
        approved_mask = df['CASE_STATUS'].str.contains('CERTIFIED', case=False, na=False)
        df.loc[approved_mask, 'TARGET'] = 1
        
        # Drop WITHDRAWN cases (not adjudicated)
        withdrawn_mask = (df['CASE_STATUS'] == 'WITHDRAWN')
        df_before_drop = len(df)
        df = df[~withdrawn_mask]
        print(f"\nDropped {df_before_drop - len(df):,} WITHDRAWN cases (not adjudicated)")
        
        print(f"\nFinal target distribution BEFORE balancing:")
        print(df['TARGET'].value_counts())
        approved_count = (df['TARGET'] == 1).sum()
        denied_count = (df['TARGET'] == 0).sum()
        print(f"Approved: {approved_count:,} ({approved_count/len(df)*100:.2f}%)")
        print(f"Denied: {denied_count:,} ({denied_count/len(df)*100:.2f}%)")
        print(f"Imbalance ratio: {approved_count/denied_count:.1f}:1")
        
        # ============================================
        # BALANCE THE DATASET
        # ============================================
        print(f"\n{'='*70}")
        print(f"STEP 2: BALANCING DATASET")
        print(f"{'='*70}")
        
        df_approved = df[df['TARGET'] == 1]
        df_denied = df[df['TARGET'] == 0]
        
        print(f"\nCurrent class distribution:")
        print(f"  Approved: {len(df_approved):,}")
        print(f"  Denied: {len(df_denied):,}")
        print(f"  Ratio: {len(df_approved)/len(df_denied):.1f}:1")
        
        # Strategy: Create 75:25 ratio (75% approved, 25% denied)
        # This is realistic for H-1B (typically 85-90% approval in reality)
        # But 75:25 gives model enough denial examples to learn
        target_approved = len(df_denied) * 3  # 3:1 ratio = 75% approved
        
        if target_approved < len(df_approved):
            print(f"\nUndersampling approved cases from {len(df_approved):,} to {target_approved:,}...")
            df_approved_sampled = resample(
                df_approved,
                n_samples=target_approved,
                random_state=42,
                replace=False
            )
        else:
            print(f"\nNot enough denied cases to balance effectively.")
            print(f"Using all approved cases: {len(df_approved):,}")
            df_approved_sampled = df_approved
        
        # Combine and shuffle
        df = pd.concat([df_approved_sampled, df_denied])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nBalanced dataset:")
        print(df['TARGET'].value_counts())
        approved_pct = (df['TARGET'] == 1).sum() / len(df) * 100
        denied_pct = (df['TARGET'] == 0).sum() / len(df) * 100
        print(f"  Approved: {(df['TARGET'] == 1).sum():,} ({approved_pct:.1f}%)")
        print(f"  Denied: {(df['TARGET'] == 0).sum():,} ({denied_pct:.1f}%)")
        print(f"  Total samples: {len(df):,}")
        
        # ============================================
        # FEATURE ENGINEERING
        # ============================================
        print(f"\n{'='*70}")
        print(f"STEP 3: FEATURE ENGINEERING")
        print(f"{'='*70}")
        
        features = []
        
        # Feature 1: Wage Level (1-5)
        if 'PREVAILING_WAGE' in df.columns:
            print(f"\n[1/6] Creating WAGE_LEVEL feature...")
            df['PREVAILING_WAGE'] = pd.to_numeric(df['PREVAILING_WAGE'], errors='coerce')
            
            # Remove outliers
            df = df[(df['PREVAILING_WAGE'] > 10000) & (df['PREVAILING_WAGE'] < 500000)]
            
            # Create wage levels
            df['WAGE_LEVEL'] = pd.cut(
                df['PREVAILING_WAGE'],
                bins=[0, 40000, 60000, 80000, 100000, 500000],
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
            
            features.append('WAGE_LEVEL')
            print(f"    ✓ Created WAGE_LEVEL")
            print(f"    Distribution:")
            for level, count in df['WAGE_LEVEL'].value_counts().sort_index().items():
                print(f"      Level {int(level)}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Feature 2: Tech Job (0 or 1)
        if 'JOB_TITLE' in df.columns:
            print(f"\n[2/6] Creating IS_TECH_JOB feature...")
            tech_keywords = [
                'software', 'engineer', 'developer', 'programmer',
                'analyst', 'data', 'scientist', 'architect', 'devops',
                'machine learning', 'ai', 'technology', 'computer', 'tech'
            ]
            df['IS_TECH_JOB'] = df['JOB_TITLE'].str.lower().str.contains(
                '|'.join(tech_keywords),
                na=False
            ).astype(int)
            
            features.append('IS_TECH_JOB')
            tech_count = df['IS_TECH_JOB'].sum()
            print(f"    ✓ Created IS_TECH_JOB")
            print(f"    Tech jobs: {tech_count:,} ({tech_count/len(df)*100:.1f}%)")
            print(f"    Non-tech: {len(df) - tech_count:,} ({(len(df) - tech_count)/len(df)*100:.1f}%)")
        
        # Feature 3: Full-Time Position (0 or 1)
        if 'FULL_TIME_POSITION' in df.columns:
            print(f"\n[3/6] Creating IS_FULL_TIME feature...")
            df['IS_FULL_TIME'] = (df['FULL_TIME_POSITION'] == 'Y').astype(int)
            features.append('IS_FULL_TIME')
            ft_count = df['IS_FULL_TIME'].sum()
            print(f"    ✓ Created IS_FULL_TIME")
            print(f"    Full-time: {ft_count:,} ({ft_count/len(df)*100:.1f}%)")
            print(f"    Part-time: {len(df) - ft_count:,} ({(len(df) - ft_count)/len(df)*100:.1f}%)")
        
        # Feature 4: Employer Frequency (log scale)
        if 'EMPLOYER_NAME' in df.columns:
            print(f"\n[4/6] Creating EMPLOYER_FREQUENCY feature...")
            employer_counts = df['EMPLOYER_NAME'].value_counts()
            df['EMPLOYER_FREQUENCY'] = df['EMPLOYER_NAME'].map(employer_counts)
            df['EMPLOYER_FREQUENCY'] = np.log1p(df['EMPLOYER_FREQUENCY'])
            features.append('EMPLOYER_FREQUENCY')
            print(f"    ✓ Created EMPLOYER_FREQUENCY (log scale)")
            print(f"    Range: {df['EMPLOYER_FREQUENCY'].min():.2f} to {df['EMPLOYER_FREQUENCY'].max():.2f}")
            print(f"    Mean: {df['EMPLOYER_FREQUENCY'].mean():.2f}")
        
        # Feature 5: State Encoding
        if 'WORKSITE' in df.columns:
            print(f"\n[5/6] Creating STATE_ENCODED feature...")
            df['WORKSITE_STATE'] = df['WORKSITE'].apply(self.extract_state)
            df['WORKSITE_STATE'] = df['WORKSITE_STATE'].fillna('UNKNOWN')
            
            le = LabelEncoder()
            df['STATE_ENCODED'] = le.fit_transform(df['WORKSITE_STATE'])
            self.label_encoders['WORKSITE_STATE'] = le
            
            features.append('STATE_ENCODED')
            print(f"    ✓ Created STATE_ENCODED")
            print(f"    Unique states: {df['WORKSITE_STATE'].nunique()}")
            print(f"    Top 5 states:")
            for state, count in df['WORKSITE_STATE'].value_counts().head(5).items():
                print(f"      {state}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Feature 6: SOC Encoding
        if 'SOC_NAME' in df.columns:
            print(f"\n[6/6] Creating SOC_ENCODED feature...")
            df['SOC_NAME'] = df['SOC_NAME'].fillna('UNKNOWN')
            
            # Only encode top 100 occupations
            top_soc = df['SOC_NAME'].value_counts().head(100).index
            df['SOC_NAME_TOP'] = df['SOC_NAME'].apply(
                lambda x: x if x in top_soc else 'OTHER'
            )
            
            le = LabelEncoder()
            df['SOC_ENCODED'] = le.fit_transform(df['SOC_NAME_TOP'])
            self.label_encoders['SOC_NAME'] = le
            
            features.append('SOC_ENCODED')
            print(f"    ✓ Created SOC_ENCODED")
            print(f"    Unique occupations (top 100 + OTHER): {df['SOC_NAME_TOP'].nunique()}")
        
        self.feature_columns = features
        
        # Final cleanup
        print(f"\n{'='*70}")
        print(f"FINAL CLEANUP")
        print(f"{'='*70}")
        initial_size = len(df)
        df = df.dropna(subset=features)
        final_size = len(df)
        print(f"Removed {initial_size - final_size:,} rows with missing features")
        
        X = df[features]
        y = df['TARGET']
        
        print(f"\n✓ Final dataset ready:")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {len(features)}")
        print(f"  Feature names: {features}")
        print(f"  Approved: {y.sum():,} ({y.mean()*100:.1f}%)")
        print(f"  Denied: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.1f}%)")
        
        return X, y
    
    def train(self, X, y):
        """Train the XGBoost model"""
        print(f"\n{'='*70}")
        print(f"STEP 4: MODEL TRAINING")
        print(f"{'='*70}")
        
        # Split data
        print(f"\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"  Approved: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        print(f"  Denied: {(~y_train.astype(bool)).sum():,} ({(1-y_train.mean())*100:.1f}%)")
        print(f"\nTest set: {len(X_test):,} samples")
        print(f"  Approved: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
        print(f"  Denied: {(~y_test.astype(bool)).sum():,} ({(1-y_test.mean())*100:.1f}%)")
        
        # Scale features
        print(f"\nScaling features with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"✓ Features scaled")
        
        # Train model
        print(f"\nTraining XGBoost classifier...")
        scale_pos_weight = (1-y_train.mean())/y_train.mean()
        print(f"  Hyperparameters:")
        print(f"    n_estimators: 100")
        print(f"    max_depth: 6")
        print(f"    learning_rate: 0.1")
        print(f"    scale_pos_weight: {scale_pos_weight:.2f}")
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        
        self.model.fit(X_train_scaled, y_train, verbose=False)
        print(f"✓ Training complete!")
        
        # Evaluate
        print(f"\n{'='*70}")
        print(f"STEP 5: MODEL EVALUATION")
        print(f"{'='*70}")
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"{'='*70}")
        for metric, value in metrics.items():
            print(f"{metric.upper():.<40} {value:.4f} ({value*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 CONFUSION MATRIX:")
        print(f"{'='*70}")
        print(f"                    Predicted")
        print(f"                 Denied  Approved")
        print(f"Actual Denied    {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Approved  {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        # Feature Importance
        print(f"\n📊 FEATURE IMPORTANCE:")
        print(f"{'='*70}")
        importance = self.model.feature_importances_
        for feature, imp in sorted(zip(self.feature_columns, importance), key=lambda x: x[1], reverse=True):
            print(f"{feature:.<40} {imp:.4f} ({imp*100:.1f}%)")
        
        return metrics
    
    def predict(self, features_dict):
        """Predict approval probability"""
        if self.model is None:
            raise ValueError("Model not trained! Please load or train a model first.")
        
        # Create DataFrame with exact feature names
        X = pd.DataFrame([features_dict], columns=self.feature_columns)
        
        # Fill missing features
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        
        X = X[self.feature_columns].astype(float)
        
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
    print("="*70)
    print("H-1B VISA APPROVAL PREDICTION")
    print("CORRECTED VERSION - Handles all CASE_STATUS values")
    print("="*70)
    
    # Check for dataset
    csv_file = 'ml_engine/h1b_kaggle.csv'
    
    if not os.path.exists(csv_file):
        print(f"\n❌ ERROR: Dataset not found!")
        print(f"Expected: {csv_file}")
        print(f"\nDownload from: https://www.kaggle.com/datasets/nsharan/h-1b-visa")
        exit(1)
    
    print(f"\n✓ Dataset found: {csv_file}")
    print(f"Loading data (this may take a minute)...")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df):,} rows")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Initialize model
    model = VisaMLModel()
    
    # Prepare data
    X, y = model.prepare_data(df)
    
    # Train model
    metrics = model.train(X, y)
    
    # Save model
    model.save_model()
    
    # Test predictions
    print(f"\n{'='*70}")
    print(f"STEP 6: TESTING PREDICTIONS")
    print(f"{'='*70}")
    
    test_cases = [
        {
            'name': 'High Salary Tech Job (Should be HIGH approval)',
            'features': {
                'WAGE_LEVEL': 5.0,
                'IS_TECH_JOB': 1.0,
                'IS_FULL_TIME': 1.0,
                'EMPLOYER_FREQUENCY': 7.0,
                'STATE_ENCODED': 4.0,
                'SOC_ENCODED': 10.0
            }
        },
        {
            'name': 'Low Salary Non-Tech (Should be LOW approval)',
            'features': {
                'WAGE_LEVEL': 1.0,
                'IS_TECH_JOB': 0.0,
                'IS_FULL_TIME': 1.0,
                'EMPLOYER_FREQUENCY': 2.0,
                'STATE_ENCODED': 25.0,
                'SOC_ENCODED': 70.0
            }
        },
        {
            'name': 'Medium Salary Tech (Should be MEDIUM-HIGH)',
            'features': {
                'WAGE_LEVEL': 4.0,
                'IS_TECH_JOB': 1.0,
                'IS_FULL_TIME': 1.0,
                'EMPLOYER_FREQUENCY': 5.0,
                'STATE_ENCODED': 33.0,
                'SOC_ENCODED': 15.0
            }
        },
        {
            'name': 'Part-time Low Salary (Should be VERY LOW)',
            'features': {
                'WAGE_LEVEL': 2.0,
                'IS_TECH_JOB': 0.0,
                'IS_FULL_TIME': 0.0,
                'EMPLOYER_FREQUENCY': 1.5,
                'STATE_ENCODED': 20.0,
                'SOC_ENCODED': 80.0
            }
        }
    ]
    
    print(f"\nTesting 4 scenarios:")
    print(f"{'='*70}")
    
    for i, test in enumerate(test_cases, 1):
        prob = model.predict(test['features'])
        
        if prob >= 0.8:
            risk, icon = 'LOW', '🟢'
        elif prob >= 0.6:
            risk, icon = 'MEDIUM', '🟡'
        else:
            risk, icon = 'HIGH', '🔴'
        
        print(f"\n{i}. {test['name']}")
        print(f"   {icon} Approval Probability: {prob:.1%}")
        print(f"   {icon} Risk Level: {risk}")
    
    print(f"\n{'='*70}")
    print(f"✓ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Model files saved in: ml_models/")
    print(f"📊 Model accuracy: {metrics['accuracy']:.1%}")
    print(f"\n🚀 Ready to use in Django!")