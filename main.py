"""
Cleveland Heart Disease Prediction with Logistic Regression
Objective: Implement logistic regression to predict heart disease using the validated Cleveland dataset
and document the ML process step-by-step.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
import kagglehub
import os

def download_dataset():
    """
    Step 1: Dataset Download and Initial Setup
    What: Download the Cleveland Heart Disease dataset from Kaggle
    Why: Need real-world medical data with validated clinical relationships
    How: Using kagglehub to download the validated UCI heart disease dataset
    """
    print("STEP 1: DATASET DOWNLOAD")
    print("-" * 40)
    
    try:
        # Download the Cleveland Heart Disease dataset (much better than synthetic data)
        path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
        print("Dataset downloaded successfully!")
        print(f"Path: {path}")
        
        # Find the CSV file
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            dataset_path = os.path.join(path, csv_files[0])
            print(f"Found dataset file: {csv_files[0]}")
            return dataset_path
        else:
            print("ERROR: No CSV file found in downloaded dataset")
            return None
            
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        return None

def explore_dataset(df):
    """
    Step 2: Dataset Exploration and Analysis
    What: Explore the Cleveland Heart Disease dataset structure and characteristics
    Why: Understanding data helps in proper preprocessing and feature selection
    How: Using pandas methods to analyze shape, types, distributions, and missing values
    """
    print("\nSTEP 2: DATASET EXPLORATION")
    print("-" * 40)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Features: {df.shape[1]} columns, {df.shape[0]} rows")
    
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found!")
    
    print("\nTarget Variable Distribution:")
    # Cleveland dataset uses 'num' as target (0 = no disease, 1-4 = disease presence)
    target_column = None
    if 'num' in df.columns:
        target_column = 'num'
        # Convert multi-class to binary (0 = no disease, 1+ = disease)
        df['target'] = (df['num'] > 0).astype(int)
        target_column = 'target'
    elif 'target' in df.columns:
        target_column = 'target'
    elif 'Heart Disease Risk' in df.columns:
        target_column = 'Heart Disease Risk'
    
    if target_column:
        target_counts = df[target_column].value_counts()
        print(f"Target column: {target_column}")
        print(target_counts)
        print(f"Class Balance: {target_counts[1]/len(df)*100:.1f}% positive cases")
        
        # Show original 'num' distribution if available
        if 'num' in df.columns and target_column != 'num':
            print(f"\nOriginal 'num' distribution (0=no disease, 1-4=disease severity):")
            print(df['num'].value_counts().sort_index())
    else:
        print("Target variable not found!")
    
    print("\nColumn Information:")
    print("Available columns:", list(df.columns))
    
    print(f"\nData Types:")
    print(f"Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"Numerical: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
    
    # Show some sample data
    print(f"\nSample data (first 5 rows):")
    print(df.head())
    
    return df

def preprocess_data(df):
    """
    Step 3: Data Preprocessing and Feature Engineering
    What: Prepare Cleveland Heart Disease data for machine learning model training
    Why: Raw data needs cleaning and transformation for optimal model performance
    How: Handle categorical variables, missing values, create medical features, and prepare X, y splits
    """
    print("\nSTEP 3: DATA PREPROCESSING")
    print("-" * 40)
    
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Identify the target variable (should be 'target' from explore step)
    target_column = None
    if 'target' in df_processed.columns:
        target_column = 'target'
    elif 'num' in df_processed.columns:
        # Convert multi-class to binary if not done already
        df_processed['target'] = (df_processed['num'] > 0).astype(int)
        target_column = 'target'
    else:
        print("ERROR: Cannot find target variable")
        return None, None, None
    
    print(f"Target variable: {target_column}")
    
    # Remove non-predictive columns
    columns_to_remove = ['id', 'origin', 'num', 'dataset']  # Keep original num out of features
    for col in columns_to_remove:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
            print(f"Removed non-predictive column: {col}")
    
    # HANDLE MISSING VALUES FIRST
    print("Handling missing values...")
    
    # Show missing value summary
    missing_summary = df_processed.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if len(missing_cols) > 0:
        print("Missing values found:")
        for col, count in missing_cols.items():
            pct = (count / len(df_processed)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Handle missing values strategically
    # For numeric columns with missing values
    numeric_cols_with_missing = []
    for col in ['trestbps', 'chol', 'thalch', 'oldpeak']:
        if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
            numeric_cols_with_missing.append(col)
    
    if numeric_cols_with_missing:
        print(f"Imputing numeric columns: {numeric_cols_with_missing}")
        numeric_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols_with_missing] = numeric_imputer.fit_transform(df_processed[numeric_cols_with_missing])
    
    # For categorical columns with missing values
    categorical_cols_with_missing = []
    for col in ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
            categorical_cols_with_missing.append(col)
    
    if categorical_cols_with_missing:
        print(f"Imputing categorical columns: {categorical_cols_with_missing}")
        # Use most frequent for categorical
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols_with_missing] = categorical_imputer.fit_transform(df_processed[categorical_cols_with_missing])
    
    # Handle 'ca' (number of vessels) - improved imputation
    if 'ca' in df_processed.columns:
        df_processed['ca'] = df_processed['ca'].astype(float)
        if df_processed['ca'].isnull().sum() > 0:
            # Smart imputation: high-risk patients more likely to have blocked vessels
            if 'exang' in df_processed.columns and 'oldpeak' in df_processed.columns:
                high_risk_mask = (df_processed['exang'] == True) & (df_processed['oldpeak'] > 1.0)
                df_processed.loc[df_processed['ca'].isnull() & high_risk_mask, 'ca'] = 1.0
                df_processed.loc[df_processed['ca'].isnull() & ~high_risk_mask, 'ca'] = 0.0
            else:
                df_processed['ca'].fillna(0, inplace=True)
            print("Improved ca (vessels) imputation")
    
    print("Missing value handling completed")
    
    # CLEVELAND DATASET SPECIFIC FEATURE ENGINEERING
    print("Creating Cleveland-specific medical features...")
    
    # Age Risk Groups (medically validated thresholds)
    if 'age' in df_processed.columns:
        df_processed['age_high_risk'] = (df_processed['age'] >= 65).astype(int)
        df_processed['age_medium_risk'] = ((df_processed['age'] >= 45) & (df_processed['age'] < 65)).astype(int)
        print("Age risk groups created")
    
    # Resting Blood Pressure Categories
    if 'trestbps' in df_processed.columns:
        df_processed['hypertension_stage1'] = ((df_processed['trestbps'] >= 130) & (df_processed['trestbps'] < 140)).astype(int)
        df_processed['hypertension_stage2'] = (df_processed['trestbps'] >= 140).astype(int)
        df_processed['hypotension'] = (df_processed['trestbps'] < 90).astype(int)
        print("Blood pressure categories created")
    
    # Cholesterol Risk Levels (mg/dl medical thresholds)
    if 'chol' in df_processed.columns:
        df_processed['high_cholesterol'] = (df_processed['chol'] >= 240).astype(int)
        df_processed['borderline_cholesterol'] = ((df_processed['chol'] >= 200) & (df_processed['chol'] < 240)).astype(int)
        print("Cholesterol risk levels created")
    
    # Maximum Heart Rate Categories (use correct column name)
    if 'thalch' in df_processed.columns:
        # Age-adjusted max heart rate (220 - age is theoretical max)
        if 'age' in df_processed.columns:
            df_processed['heart_rate_reserve'] = df_processed['thalch'] / (220 - df_processed['age'])
            df_processed['low_heart_rate_reserve'] = (df_processed['heart_rate_reserve'] < 0.6).astype(int)
        df_processed['low_max_heart_rate'] = (df_processed['thalch'] < 120).astype(int)
        print("Heart rate features created")
    
    # ST Depression Risk (oldpeak)
    if 'oldpeak' in df_processed.columns:
        df_processed['significant_st_depression'] = (df_processed['oldpeak'] >= 2.0).astype(int)
        df_processed['mild_st_depression'] = ((df_processed['oldpeak'] >= 1.0) & (df_processed['oldpeak'] < 2.0)).astype(int)
        print("ST depression features created")
    
    # Chest Pain Binary Features (more interpretable than risk scores)
    if 'cp' in df_processed.columns:
        # Create binary indicators instead of risk scores to avoid model confusion
        print("Creating chest pain binary features...")
        
        # Create high-risk indicator (asymptomatic patients have highest risk)
        df_processed['cp_high_risk'] = (df_processed['cp'] == 'asymptomatic').astype(int)
        print("Chest pain high-risk indicator created")
    
    # Major Vessels Blocked (ca)
    if 'ca' in df_processed.columns:
        df_processed['multiple_vessel_disease'] = (df_processed['ca'] >= 2).astype(int)
        df_processed['single_vessel_disease'] = (df_processed['ca'] == 1).astype(int)
        print("Vessel disease features created")
    
    # Composite Risk Scores
    # High-risk binary features
    binary_risk_features = ['fbs', 'exang']  # fasting blood sugar, exercise angina
    available_binary_risk = [col for col in binary_risk_features if col in df_processed.columns]
    
    # Add our engineered high-risk features (updated to include chest pain)
    engineered_risk_features = ['age_high_risk', 'hypertension_stage2', 'high_cholesterol', 
                               'low_heart_rate_reserve', 'significant_st_depression', 'multiple_vessel_disease', 'cp_high_risk']
    available_engineered_risk = [col for col in engineered_risk_features if col in df_processed.columns]
    
    all_risk_factors = available_binary_risk + available_engineered_risk
    if all_risk_factors:
        df_processed['total_risk_score'] = df_processed[all_risk_factors].sum(axis=1)
        df_processed['high_risk_patient'] = (df_processed['total_risk_score'] >= 3).astype(int)
        print(f"Composite risk score created from: {all_risk_factors}")
    
    # Handle categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from categorical cols if present
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    print(f"Categorical columns for encoding: {categorical_cols}")
    
    # Handle remaining categorical variables with one-hot encoding
    if categorical_cols:
        print("Applying One-Hot Encoding to categorical variables...")
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        print(f"Shape after encoding: {df_processed.shape}")
    
    # Final check for any remaining missing values
    final_missing = df_processed.isnull().sum().sum()
    if final_missing > 0:
        print(f"WARNING: {final_missing} missing values remain!")
        # Drop rows with remaining missing values as last resort
        df_processed = df_processed.dropna()
        print(f"Dropped rows with missing values. New shape: {df_processed.shape}")
    
    # Prepare features (X) and target (y)
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]
    
    print(f"Final preprocessing results:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # Show the engineered features
    engineered_features = [col for col in X.columns if any(keyword in col for keyword in 
                          ['_risk', '_score', 'hypertension', 'cholesterol', 'heart_rate', 'st_depression', 
                           'vessel', 'high_', 'low_', 'borderline_', 'significant_'])]
    if engineered_features:
        print(f"Engineered medical features: {engineered_features}")
    
    return X, y, df_processed

def split_and_scale_data(X, y):
    """
    Step 4: Train-Test Split and Feature Scaling
    What: Split data into training and testing sets, then scale features
    Why: Need separate test set for unbiased evaluation, scaling ensures all features contribute equally
    How: Stratified split to maintain class balance, StandardScaler for normalization
    """
    print("\nSTEP 4: TRAIN-TEST SPLIT & SCALING")
    print("-" * 40)
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Feature scaling completed")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic_regression(X_train, y_train):
    """
    Step 5: Logistic Regression Model Training
    What: Train logistic regression model on the prepared training data
    Why: Logistic regression is ideal for binary classification and provides interpretable results
    How: Using scikit-learn's LogisticRegression with optimized parameters and cross-validation
    """
    print("\nSTEP 5: LOGISTIC REGRESSION TRAINING")
    print("-" * 40)
    
    # Calculate class distribution for information
    class_counts = pd.Series(y_train).value_counts()
    class_ratio = class_counts[0] / class_counts[1] if 1 in class_counts else 1
    print(f"Training class distribution: {class_counts.to_dict()}")
    print(f"Class imbalance ratio (neg/pos): {class_ratio:.2f}")
    
    # FIXED: Optimized model configuration for better performance
    
    # Try different regularization strengths
    C_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_score = 0
    best_C = 1.0
    
    print("Finding optimal regularization parameter...")
    for C in C_values:
        model_temp = LogisticRegression(
            solver='lbfgs',  # Better solver for small datasets
            C=C,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # 5-fold cross validation
        cv_scores = cross_val_score(model_temp, X_train, y_train, cv=5, scoring='f1')
        avg_score = cv_scores.mean()
        
        print(f"  C={C}: F1-Score = {avg_score:.3f} (±{cv_scores.std():.3f})")
        
        if avg_score > best_score:
            best_score = avg_score
            best_C = C
    
    print(f"Best C value: {best_C} with F1-Score: {best_score:.3f}")
    
    # Train final model with best parameters
    model = LogisticRegression(
        solver='lbfgs',
        C=best_C,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    print("\nModel Configuration:")
    print(f"  Solver: {model.solver}")
    print(f"  Regularization (C): {model.C}")
    print(f"  Class weight: {model.class_weight}")
    print(f"  Max iterations: {model.max_iter}")
    
    print("\nTraining final model...")
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    return model

def make_predictions(model, X_test):
    """
    Step 6: Generate Predictions and Probability Scores
    What: Use trained model to make predictions on test set
    Why: Need predictions to evaluate model performance
    How: Generate both binary predictions and probability scores
    """
    print("\nSTEP 6: PREDICTION GENERATION")
    print("-" * 40)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"Predictions generated for {len(y_pred)} samples")
    print(f"Predicted positive cases: {sum(y_pred)}")
    print(f"Average predicted probability: {y_pred_proba.mean():.3f}")
    
    return y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba):
    """
    Step 7: Model Performance Evaluation
    What: Calculate comprehensive performance metrics for the model
    Why: Need multiple metrics to understand model strengths and weaknesses
    How: Calculate accuracy, precision, recall, F1-score, and create confusion matrix
    """
    print("\nSTEP 7: MODEL EVALUATION")
    print("-" * 40)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("PERFORMANCE METRICS:")
    print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nCONFUSION MATRIX:")
    print(f"              Predicted")
    print(f"            0      1")
    print(f"Actual  0  {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"        1  {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    return accuracy, precision, recall, f1, cm

def analyze_feature_importance(model, feature_names):
    """
    Step 8: Feature Importance and Model Interpretation
    What: Analyze and interpret the trained model coefficients
    Why: Understanding which features drive predictions helps validate model and guide medical decisions
    How: Extract and rank logistic regression coefficients with medical interpretation
    """
    print("\nSTEP 8: FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    # Get model coefficients
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    print(f"Model Intercept: {intercept:.4f}")
    print(f"Number of features: {len(coefficients)}")
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'odds_ratio': np.exp(coefficients)  # Convert to odds ratios for medical interpretation
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 70)
    print(f"{'Feature':30s} {'Coefficient':>12s} {'Odds Ratio':>12s} {'Impact':>12s}")
    print("-" * 70)
    
    for i, row in feature_importance.head(15).iterrows():
        direction = "Increases" if row['coefficient'] > 0 else "Decreases"
        odds_ratio = row['odds_ratio']
        
        # Interpret odds ratio
        if odds_ratio > 1:
            odds_interpretation = f"{odds_ratio:.2f}x risk"
        else:
            odds_interpretation = f"{1/odds_ratio:.2f}x protect"
            
        print(f"{row['feature']:30s} {row['coefficient']:>10.3f}   {odds_ratio:>10.3f}   {direction:>10s}")
    
    print(f"\n" + "=" * 70)
    print("MEDICAL INTERPRETATION:")
    print("=" * 70)
    
    # Group features by medical category for better interpretation
    categories = {
        'cardiovascular': ['trestbps', 'thalach', 'heart_rate', 'hypertension', 'hypotension'],
        'cardiac_function': ['oldpeak', 'st_depression', 'slope', 'exang'],
        'metabolic': ['chol', 'cholesterol', 'fbs'],
        'vascular': ['ca', 'vessel', 'thal'],
        'symptoms': ['cp', 'chest_pain', 'angina'],
        'demographic': ['age', 'sex'],
        'engineered': ['_risk', '_score', 'total_risk', 'high_risk_patient']
    }
    
    for category, keywords in categories.items():
        relevant_features = []
        for feature in feature_importance.head(10)['feature']:
            if any(keyword.lower().replace(' ', '_') in feature.lower() for keyword in keywords):
                relevant_features.append(feature)
        
        if relevant_features:
            print(f"\n{category.upper()} FACTORS:")
            for feature in relevant_features:
                row = feature_importance[feature_importance['feature'] == feature].iloc[0]
                direction = "increases" if row['coefficient'] > 0 else "decreases"
                print(f"  • {feature}: {direction} heart disease risk")
    
    # Medical insights
    top_5_features = feature_importance.head(5)
    print(f"\nKEY MEDICAL INSIGHTS:")
    print(f"• Most predictive factor: {top_5_features.iloc[0]['feature']}")
    
    positive_factors = feature_importance[feature_importance['coefficient'] > 0].head(3)
    negative_factors = feature_importance[feature_importance['coefficient'] < 0].head(3)
    
    if not positive_factors.empty:
        print(f"• Top risk increasers: {', '.join(positive_factors['feature'].tolist())}")
    if not negative_factors.empty:
        print(f"• Top protective factors: {', '.join(negative_factors['feature'].tolist())}")
    
    print(f"\nCLINICAL INTERPRETATION:")
    print(f"• This model uses the validated Cleveland Heart Disease dataset")
    print(f"• Features reflect real cardiovascular risk factors")
    print(f"• Results should align with established medical knowledge")
    
    return feature_importance

def create_visualizations(cm, feature_importance):
    """
    Create visualizations for confusion matrix and feature importance
    """
    print("\nCREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Feature Importance Plot
    top_features = feature_importance.head(10)
    colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
    ax2.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'])
    ax2.set_xlabel('Coefficient Value')
    ax2.set_title('Top 10 Feature Importance')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heart_disease_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'heart_disease_prediction_results.png'")

def analyze_data_quality(df):
    """
    Analyze Cleveland Heart Disease dataset quality and relationships
    """
    print("\nCLEVELAND DATASET QUALITY ANALYSIS")
    print("-" * 40)
    
    target_col = 'target' if 'target' in df.columns else 'num'
    
    # Show original distribution if num exists
    if 'num' in df.columns:
        print("Original disease severity distribution:")
        print("0 = No disease, 1-4 = Disease severity levels")
        print(df['num'].value_counts().sort_index())
        print()
    
    # Analyze Age vs Heart Disease
    if 'age' in df.columns:
        print("Age vs Heart Disease:")
        age_analysis = df.groupby(pd.cut(df['age'], bins=8))[target_col].agg(['mean', 'count'])
        print(age_analysis)
    
    # Analyze Chest Pain vs Heart Disease
    if 'cp' in df.columns:
        print(f"\nChest Pain Type vs Heart Disease:")
        cp_labels = {0: 'Asymptomatic', 1: 'Atypical Angina', 2: 'Non-anginal', 3: 'Typical Angina'}
        df_temp = df.copy()
        df_temp['cp_label'] = df_temp['cp'].map(cp_labels)
        cp_analysis = df_temp.groupby('cp_label')[target_col].agg(['mean', 'count'])
        print(cp_analysis)
    
    # Analyze Cholesterol vs Heart Disease
    if 'chol' in df.columns:
        print(f"\nCholesterol vs Heart Disease:")
        # Exclude zero values (missing data)
        chol_data = df[df['chol'] > 0]
        if len(chol_data) > 0:
            chol_analysis = chol_data.groupby(pd.cut(chol_data['chol'], bins=5))[target_col].agg(['mean', 'count'])
            print(chol_analysis)
        print(f"Zero cholesterol values (missing): {(df['chol'] == 0).sum()}")
    
    # Analyze Max Heart Rate vs Heart Disease
    if 'thalach' in df.columns:
        print(f"\nMax Heart Rate vs Heart Disease:")
        hr_analysis = df.groupby(pd.cut(df['thalach'], bins=5))[target_col].agg(['mean', 'count'])
        print(hr_analysis)
    
    # Key clinical correlations
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if target_col in numeric_cols:
        print(f"\nTop Correlations with Heart Disease:")
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        print(correlations.head(10))
    
    # Sample of key medical data
    key_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', target_col]
    available_cols = [col for col in key_cols if col in df.columns]
    print(f"\nSample of key medical data:")
    print(df[available_cols].head(10))

def main():
    """
    Main function to execute the complete Cleveland Heart Disease prediction pipeline
    """
    print("CLEVELAND HEART DISEASE PREDICTION WITH LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Step 1: Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        print("ERROR: Cannot proceed without dataset")
        return
    
    # Load the dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully: {df.shape}")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return
    
    # Step 2: Explore dataset
    df = explore_dataset(df)
    
    # Step 3: Preprocess data
    X, y, df_processed = preprocess_data(df)
    if X is None:
        print("ERROR: Data preprocessing failed")
        return
    
    # Step 4: Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Step 5: Train model
    model = train_logistic_regression(X_train_scaled, y_train)
    
    # Step 6: Make predictions
    y_pred, y_pred_proba = make_predictions(model, X_test_scaled)
    
    # Step 7: Evaluate model
    accuracy, precision, recall, f1, cm = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Step 8: Analyze feature importance
    feature_importance = analyze_feature_importance(model, X.columns)
    
    # Create visualizations
    create_visualizations(cm, feature_importance)
    
    # Final summary
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model Accuracy: {accuracy:.1%}")
    print(f"Dataset Size: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Most Important Feature: {feature_importance.iloc[0]['feature']}")
    print("Results visualized and saved")

    # Analyze data quality
    analyze_data_quality(df)

if __name__ == "__main__":
    main()
