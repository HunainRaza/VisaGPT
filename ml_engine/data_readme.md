# Dataset Information

## H-1B Visa Dataset

This project uses the **H-1B Visa Petitions dataset** for training the machine learning model.

### Dataset Details
- **Source:** Kaggle - H-1B Visa Petitions 2011-2016
- **File:** `h1b_kaggle.csv`
- **Size:** ~469 MB
- **Records:** 3+ million visa applications
- **Features:** Case status, employer, job title, wage, location, etc.

### Download Instructions

1. **Download the dataset from Kaggle:**
   - Visit: https://www.kaggle.com/datasets/nsharan/h-1b-visa
   - Download: `h1b_kaggle.csv`

2. **Place the file:**
   ```
   ml_engine/h1b_kaggle.csv
   ```

3. **Verify placement:**
   ```bash
   ls ml_engine/h1b_kaggle.csv
   ```

### Training the Model

Once the dataset is in place:

```bash
python ml_engine/train_model.py
```

This will:
- Load and clean 3M+ records
- Engineer features
- Train XGBoost classifier
- Save model to `ml_models/`
- Output: 93.95% accuracy

### Pre-trained Models

If you don't want to train from scratch, the pre-trained models are already included in `ml_models/`:
- `visa_model.pkl` - Trained XGBoost classifier
- `label_encoders.pkl` - Categorical encoders
- `scaler.pkl` - Numerical scaler
- `feature_columns.pkl` - Feature configuration

### Dataset Statistics

- **Total Records:** 3,002,458
- **Certified:** 2,615,623 (87.1%)
- **Denied:** 386,835 (12.9%)
- **Time Period:** 2011-2016
- **Employers:** 236,000+
- **Job Titles:** 287,000+

### Features Used for Training

1. Job classification (tech vs non-tech)
2. Wage level (1-5 scale)
3. Full-time status
4. Employer frequency
5. State encoding
6. SOC code encoding

### Citation

If using this dataset, please cite:
```
H-1B Visa Petitions 2011-2016
Kaggle Dataset
https://www.kaggle.com/datasets/nsharan/h-1b-visa
```

---

**Note:** The CSV file is not included in this repository due to GitHub's file size limits. Download it separately from Kaggle.