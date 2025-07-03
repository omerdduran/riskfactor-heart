# Cleveland Heart Disease Prediction with Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-82.1%25-brightgreen.svg)](#performance-metrics)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.842-brightgreen.svg)](#performance-metrics)

A comprehensive machine learning project that predicts heart disease using the validated Cleveland Heart Disease dataset. This implementation achieves **82.1% accuracy** with robust feature engineering and medical interpretation.

## 🏥 Project Overview

This project implements a logistic regression model to predict heart disease presence using the famous Cleveland Heart Disease dataset from UCI. The model incorporates advanced feature engineering, missing value handling, and provides medically interpretable results that align with established cardiovascular risk factors.

### Key Features

- ✅ **High Accuracy**: 82.1% accuracy with 84.2% F1-score
- ✅ **Real Medical Data**: Uses validated Cleveland Heart Disease UCI dataset
- ✅ **Robust Preprocessing**: Handles 66% missing values in some features
- ✅ **Feature Engineering**: Creates 20+ medically meaningful features
- ✅ **Medical Interpretation**: Results align with clinical knowledge
- ✅ **Cross-validation**: Hyperparameter tuning with 5-fold CV
- ✅ **Visualization**: Confusion matrix and feature importance plots

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 82.1% |
| **Precision** | 82.2% |
| **Recall** | 86.3% |
| **F1-Score** | 84.2% |
| **AUC-ROC** | High confidence predictions |

### Confusion Matrix
```
                Predicted
              0      1
Actual   0   63     19    (77% specificity)
         1   14     88    (86% sensitivity)
```

## 🔬 Dataset Information

**Source**: Cleveland Heart Disease UCI Dataset  
**Size**: 920 patients, 16 original features  
**Target**: Binary classification (0 = No disease, 1 = Disease present)  
**Class Distribution**: 55.3% positive cases (509), 44.7% negative cases (411)

### Original Features
- `age`: Age in years
- `sex`: Gender (Male/Female)
- `cp`: Chest pain type (4 categories)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting electrocardiographic results
- `thalch`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy (0-3)
- `thal`: Thalassemia type

### Missing Data Handling
The dataset contains significant missing values that are professionally handled:
- `ca`: 66.4% missing → Imputed with most frequent
- `thal`: 52.8% missing → Imputed with most frequent  
- `slope`: 33.6% missing → Imputed with most frequent
- Other features: 3-10% missing → Median/mode imputation

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- uv (Python package manager) or pip

### Quick Start
```bash
# Clone the repository
git clone https://github.com/omerdduran/riskfactor-heart.git
cd riskfactor-heart

# Install dependencies
uv sync  # or pip install -r requirements.txt

# Run the complete analysis
uv run main.py  # or python main.py
```

### Dependencies
```toml
[project]
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "kagglehub>=0.2.0"
]
```

## 🚀 Usage

### Basic Usage
```python
# Run the complete pipeline
python main.py
```

The script will automatically:
1. Download the Cleveland Heart Disease dataset
2. Perform comprehensive data exploration
3. Handle missing values with medical-aware imputation
4. Engineer 20+ cardiovascular risk features
5. Train and optimize logistic regression model
6. Generate performance metrics and visualizations
7. Provide medical interpretation of results

### Output Files
- `heart_disease_prediction_results.png`: Confusion matrix and feature importance visualization
- Console output: Detailed step-by-step analysis and medical insights

## 🔬 Feature Engineering

The model creates **40 features** from 16 original features, including:

### Medical Risk Categories
- **Age Groups**: High risk (≥65), Medium risk (45-64), Young (<45)
- **Blood Pressure**: Hypertension Stage 1&2, Hypotension
- **Cholesterol**: High (≥240), Borderline (200-240), Normal (<200)
- **Heart Rate**: Age-adjusted heart rate reserve, low max heart rate
- **ST Depression**: Significant (≥2.0), Mild (1.0-2.0), Normal (<1.0)
- **Chest Pain**: Risk scoring (Asymptomatic=highest risk)
- **Vessel Disease**: Multiple vessels (≥2), Single vessel, None

### Composite Risk Scores
- **Total Risk Score**: Sum of major risk factors
- **High Risk Patient**: Binary indicator (≥3 risk factors)
- **Heart Rate Reserve**: Age-adjusted cardiovascular fitness

## 📈 Model Performance Analysis

### Top Predictive Features
1. **Cholesterol** (chol): Higher levels slightly decrease risk in this dataset
2. **Male Gender**: 58.9% higher odds of heart disease
3. **Exercise Angina**: 47% higher odds when present
4. **Chest Pain Risk Score**: Higher scores indicate higher risk
5. **Age**: Each year increases risk by 33.7%

### Medical Insights
- **Age Effect**: Risk increases linearly with age (28% at 30 → 70% at 65+)
- **Gender Impact**: Males have significantly higher risk
- **Exercise Tolerance**: Exercise-induced angina is a strong predictor
- **Coronary Anatomy**: Number of blocked vessels directly correlates with risk
- **ST Depression**: Significant predictor of underlying coronary disease

## 🧪 Technical Implementation

### Data Preprocessing Pipeline
1. **Missing Value Imputation**
   - Numeric features: Median imputation
   - Categorical features: Most frequent imputation
   - Medical context-aware handling

2. **Feature Engineering**
   - Medical threshold-based categorization
   - Age-adjusted calculations
   - Composite risk scoring

3. **Model Training**
   - 5-fold cross-validation for hyperparameter tuning
   - Balanced class weights for imbalanced data
   - Optimal regularization (C=0.1) found via grid search

### Model Architecture
```python
LogisticRegression(
    solver='lbfgs',
    C=0.1,                    # Optimal regularization
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    max_iter=1000
)
```

## 📚 Medical Background

### Clinical Relevance
This model uses the **gold standard** Cleveland Heart Disease dataset, collected from:
- **Cleveland Clinic Foundation** (Primary source)
- **Hungarian Institute of Cardiology, Budapest**
- **University Hospital, Zurich, Switzerland**
- **University Hospital, Basel, Switzerland**

### Risk Factors Validated
The model correctly identifies established cardiovascular risk factors:
- ✅ **Age**: Progressive risk increase with age
- ✅ **Male Gender**: Higher risk in males
- ✅ **Exercise Intolerance**: Strong predictor
- ✅ **Coronary Anatomy**: Vessel involvement
- ✅ **Electrocardiographic Changes**: ST depression

## 🔍 Results Interpretation

### Clinical Decision Support
The model can assist healthcare providers by:
- **Risk Stratification**: Identifying high-risk patients
- **Feature Importance**: Understanding key risk drivers
- **Probability Scores**: Quantifying disease likelihood
- **Early Detection**: Flagging at-risk individuals

### Model Limitations
- Based on specific population (Cleveland clinic patients)
- Binary classification (presence vs. absence)
- Does not predict disease severity
- Requires clinical validation before deployment

## 📊 Visualization Examples

The model generates comprehensive visualizations:
- **Confusion Matrix**: Model performance breakdown
- **Feature Importance**: Top 15 predictive factors with odds ratios
- **Medical Categories**: Grouped analysis by medical domain
