"""
Cleveland Heart Disease Prediction with Logistic Regression
Objective: Implement logistic regression to predict heart disease using the validated Cleveland dataset
and document the ML process step-by-step.
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import kagglehub
import joblib

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def download_dataset():
    """
    Step 1: Dataset Download and Initial Setup
    What: Download the Cleveland Heart Disease dataset from Kaggle
    Why: Need real-world medical data with validated clinical relationships
    How: Using kagglehub to download the validated UCI heart disease dataset
    """
    logger.info("STEP 1: DATASET DOWNLOAD")
    
    try:
        # Download the Cleveland Heart Disease dataset (much better than synthetic data)
        path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
        logger.info("Dataset downloaded successfully!")
        logger.info(f"Path: {path}")
        
        # Find the CSV file
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            dataset_path = os.path.join(path, csv_files[0])
            logger.info(f"Found dataset file: {csv_files[0]}")
            return dataset_path
        else:
            logger.error("No CSV file found in downloaded dataset")
            return None
            
    except Exception as e:
        logger.exception(f"ERROR downloading dataset: {e}")
        return None

def explore_dataset(df):
    """
    Step 2: Dataset Exploration and Analysis
    What: Explore the Cleveland Heart Disease dataset structure and characteristics
    Why: Understanding data helps in proper preprocessing and feature selection
    How: Using pandas methods to analyze shape, types, distributions, and missing values
    """
    logger.info("STEP 2: DATASET EXPLORATION")
    
    logger.info(f"Dataset Shape: {df.shape}")
    logger.info(f"Features: {df.shape[1]} columns, {df.shape[0]} rows")
    
    logger.info("Missing Values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info("\n" + str(missing_values[missing_values > 0]))
    else:
        logger.info("No missing values found!")
    
    logger.info("Target Variable Distribution:")
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
        logger.info(f"Target column: {target_column}")
        logger.info("\n" + str(target_counts))
        logger.info(f"Class Balance: {target_counts[1]/len(df)*100:.1f}% positive cases")
        
        # Show original 'num' distribution if available
        if 'num' in df.columns and target_column != 'num':
            logger.info("Original 'num' distribution (0=no disease, 1-4=disease severity):")
            logger.info("\n" + str(df['num'].value_counts().sort_index()))
    else:
        logger.warning("Target variable not found!")
    
    logger.info("Column Information:")
    logger.info(f"Available columns: {list(df.columns)}")
    
    logger.info("Data Types:")
    logger.info(f"Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    logger.info(f"Numerical: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
    
    # Show some sample data
    logger.info("Sample data (first 5 rows):\n" + str(df.head()))
    
    return df

class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to create medically meaningful features.

    Note: This transformer must not use target information; it should
    only derive features from X to avoid leakage.
    """

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X):
        X_out = X.copy()

        # Normalize dtypes early for consistency
        if 'chol' in X_out.columns:
            # Treat zero cholesterol as missing
            with np.errstate(invalid='ignore'):
                zero_chol_count = (X_out['chol'] == 0).sum()
            if zero_chol_count > 0:
                logger.debug(f"Converting {int(zero_chol_count)} zero cholesterol values to NaN")
                X_out.loc[X_out['chol'] == 0, 'chol'] = np.nan

        # Age risk groups
        if 'age' in X_out.columns:
            X_out['age_high_risk'] = (X_out['age'] >= 65).astype(int)
            X_out['age_medium_risk'] = ((X_out['age'] >= 45) & (X_out['age'] < 65)).astype(int)

        # Blood pressure categories
        if 'trestbps' in X_out.columns:
            X_out['hypertension_stage1'] = ((X_out['trestbps'] >= 130) & (X_out['trestbps'] < 140)).astype(int)
            X_out['hypertension_stage2'] = (X_out['trestbps'] >= 140).astype(int)
            X_out['hypotension'] = (X_out['trestbps'] < 90).astype(int)

        # Cholesterol categories
        if 'chol' in X_out.columns:
            X_out['high_cholesterol'] = (X_out['chol'] >= 240).fillna(0).astype(int)
            X_out['borderline_cholesterol'] = ((X_out['chol'] >= 200) & (X_out['chol'] < 240)).fillna(0).astype(int)

        # Heart rate derived features
        if 'thalch' in X_out.columns:
            if 'age' in X_out.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    X_out['heart_rate_reserve'] = X_out['thalch'] / (220 - X_out['age'])
            X_out['low_heart_rate_reserve'] = (X_out.get('heart_rate_reserve', 1.0) < 0.6).astype(int)
            X_out['low_max_heart_rate'] = (X_out['thalch'] < 120).astype(int)

        # ST depression
        if 'oldpeak' in X_out.columns:
            X_out['significant_st_depression'] = (X_out['oldpeak'] >= 2.0).astype(int)
            X_out['mild_st_depression'] = ((X_out['oldpeak'] >= 1.0) & (X_out['oldpeak'] < 2.0)).astype(int)

        # Chest pain (string categories in some variants)
        if 'cp' in X_out.columns:
            # normalize to lowercase strings if not numeric
            if not np.issubdtype(X_out['cp'].dtype, np.number):
                cp_lower = X_out['cp'].astype(str).str.lower()
                X_out['cp_high_risk'] = (cp_lower == 'asymptomatic').astype(int)
            else:
                # If numeric (0-3), map 0: typical angina, 3: asymptomatic (UCI conv.)
                X_out['cp_high_risk'] = (X_out['cp'] == 3).astype(int)

        # Vessel disease
        if 'ca' in X_out.columns:
            with np.errstate(invalid='ignore'):
                X_out['multiple_vessel_disease'] = (X_out['ca'] >= 2).astype(int)
                X_out['single_vessel_disease'] = (X_out['ca'] == 1).astype(int)

        # Binary risk features from raw columns (ensure 0/1 ints)
        for col in ['fbs', 'exang']:
            if col in X_out.columns:
                if X_out[col].dtype == bool:
                    X_out[col] = X_out[col].astype(int)
                else:
                    # attempt to coerce to int 0/1 safely
                    X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(0).astype(int)

        # Composite score
        candidates = [
            c for c in [
                'fbs', 'exang', 'age_high_risk', 'hypertension_stage2', 'high_cholesterol',
                'low_heart_rate_reserve', 'significant_st_depression', 'multiple_vessel_disease', 'cp_high_risk'
            ] if c in X_out.columns
        ]
        if candidates:
            X_out['total_risk_score'] = X_out[candidates].sum(axis=1)
            X_out['high_risk_patient'] = (X_out['total_risk_score'] >= 3).astype(int)

        return X_out


def prepare_features_and_target(df):
    """Prepare X and y without applying any imputation/encoding to avoid leakage."""
    logger.info("STEP 3: DATA PREPARATION (no-leakage)")

    df_processed = df.copy()

    # Determine target column
    if 'target' in df_processed.columns:
        target_column = 'target'
    elif 'num' in df_processed.columns:
        df_processed['target'] = (df_processed['num'] > 0).astype(int)
        target_column = 'target'
    else:
        logger.error("Cannot find target variable ('target' or 'num')")
        return None, None

    # Drop non-predictive / leakage-prone columns
    columns_to_remove = ['id', 'origin', 'num', 'dataset']
    existing_to_drop = [c for c in columns_to_remove if c in df_processed.columns]
    if existing_to_drop:
        logger.info(f"Dropping non-predictive columns: {existing_to_drop}")
        df_processed = df_processed.drop(existing_to_drop, axis=1)

    y = df_processed[target_column].astype(int)
    X = df_processed.drop(target_column, axis=1)

    logger.info(f"Features shape: {X.shape} | Target shape: {y.shape}")
    return X, y

def split_data(X, y, test_size=0.2, seed=42):
    """Stratified train-test split only (no scaling)."""
    logger.info("STEP 4: TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    logger.info(f"Training set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def build_and_train_pipeline(X_train, y_train):
    """Build a leakage-free Pipeline with ColumnTransformer and train with LR-CV."""
    logger.info("STEP 5: MODELING (Pipeline + LogisticRegressionCV)")

    # Preprocessor: numeric and categorical branches
    numeric_processor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_processor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_processor, selector(dtype_include=np.number)),
            ("cat", categorical_processor, selector(dtype_include=object)),
        ],
        remainder="drop",
    )

    # Logistic Regression with cross-validated C selection
    clf = LogisticRegressionCV(
        Cs=[0.1, 0.5, 1.0, 2.0, 5.0],
        cv=5,
        scoring="f1",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=None,
        refit=True,
    )

    pipeline = Pipeline(steps=[
        ("feature_engineering", MedicalFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])

    class_counts = pd.Series(y_train).value_counts().to_dict()
    neg = class_counts.get(0, 0)
    pos = class_counts.get(1, 0)
    ratio = (neg / pos) if pos else float('inf')
    logger.info(f"Training class distribution: {class_counts} | Neg/Pos ratio: {ratio:.2f}")

    logger.info("Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed")

    # Best C from LogisticRegressionCV
    best_C = float(np.ravel(pipeline.named_steps['clf'].C_)[0])
    logger.info(f"Best regularization C selected by CV: {best_C}")

    return pipeline, best_C

def make_predictions(model, X_test):
    """
    Step 6: Generate Predictions and Probability Scores
    What: Use trained model to make predictions on test set
    Why: Need predictions to evaluate model performance
    How: Generate both binary predictions and probability scores
    """
    logger.info("STEP 6: PREDICTION GENERATION")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    logger.info(f"Predictions generated for {len(y_pred)} samples")
    logger.info(f"Predicted positive cases: {int(np.sum(y_pred))}")
    logger.info(f"Average predicted probability: {y_pred_proba.mean():.3f}")
    
    return y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba):
    """
    Step 7: Model Performance Evaluation
    What: Calculate comprehensive performance metrics for the model
    Why: Need multiple metrics to understand model strengths and weaknesses
    How: Calculate accuracy, precision, recall, F1-score, and create confusion matrix
    """
    logger.info("STEP 7: MODEL EVALUATION")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Additional metrics (Quick Wins)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    logger.info("PERFORMANCE METRICS:")
    logger.info(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    logger.info(f"  Precision: {precision:.3f}")
    logger.info(f"  Recall:    {recall:.3f}")
    logger.info(f"  F1-Score:  {f1:.3f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.3f}")
    logger.info(f"  PR-AUC:    {pr_auc:.3f}")
    logger.info(f"  Brier:     {brier:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion matrix computed")

    return accuracy, precision, recall, f1, cm, {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier}

def analyze_feature_importance(model, X_fit_columns):
    """
    Step 8: Feature Importance and Model Interpretation
    What: Analyze and interpret the trained model coefficients
    Why: Understanding which features drive predictions helps validate model and guide medical decisions
    How: Extract and rank logistic regression coefficients with medical interpretation
    """
def analyze_feature_importance(pipeline, X_sample):
    """Extract feature names from preprocessor and align with coefficients."""
    logger.info("STEP 8: FEATURE IMPORTANCE ANALYSIS")

    clf = pipeline.named_steps['clf']
    pre = pipeline.named_steps['preprocessor']
    # Get names after preprocessing
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        # Fallback: generate generic names
        feature_names = np.array([f"f_{i}" for i in range(clf.coef_.shape[1])])

    coefficients = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    logger.info(f"Model Intercept: {intercept:.4f} | Features: {len(coefficients)}")

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'odds_ratio': np.exp(coefficients)
    }).sort_values('abs_coefficient', ascending=False)

    # Brief console report
    head = feature_importance.head(15)
    logger.info("Top features (by |coefficient|):\n" + head[['feature', 'coefficient', 'odds_ratio']].to_string(index=False))

    return feature_importance

def create_visualizations(cm, feature_importance, y_test=None, y_proba=None):
    """
    Create visualizations for confusion matrix and feature importance
    """
    logger.info("CREATING VISUALIZATIONS")
    
    # Set up the plotting style
    os.makedirs('artifacts', exist_ok=True)
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3 if (y_test is not None and y_proba is not None) else 2, figsize=(21, 6))
    if isinstance(axes, np.ndarray):
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2] if len(axes) > 2 else None
    else:
        ax1, ax2 = axes, None
    
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

    # ROC curve (optional third panel)
    if y_test is not None and y_proba is not None and ax3 is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax3.plot(fpr, tpr, label='ROC curve')
        ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve')
        ax3.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('artifacts/heart_disease_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info("Visualizations saved under 'artifacts/'")


def save_artifacts(pipeline, feature_importance, metrics_dict):
    """Persist model, preprocessor info, feature importance, and metrics."""
    os.makedirs('artifacts', exist_ok=True)

    # Model pipeline
    model_path = os.path.join('artifacts', 'model_pipeline.joblib')
    joblib.dump(pipeline, model_path)

    # Feature importance
    fi_path = os.path.join('artifacts', 'feature_importance.csv')
    feature_importance.to_csv(fi_path, index=False)

    # Metrics with timestamp
    metrics_path = os.path.join('artifacts', 'metrics.json')
    payload = {
        **metrics_dict,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(metrics_path, 'w') as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Artifacts saved: {model_path}, {fi_path}, {metrics_path}")

def analyze_data_quality(df):
    """
    Analyze Cleveland Heart Disease dataset quality and relationships
    """
    logger.info("CLEVELAND DATASET QUALITY ANALYSIS")
    
    target_col = 'target' if 'target' in df.columns else 'num'
    
    # Show original distribution if num exists
    if 'num' in df.columns:
        logger.info("Original disease severity distribution:")
        logger.info("0 = No disease, 1-4 = Disease severity levels")
        logger.info("\n" + str(df['num'].value_counts().sort_index()))
    
    # Analyze Age vs Heart Disease
    if 'age' in df.columns:
        logger.info("Age vs Heart Disease:")
        age_analysis = df.groupby(pd.cut(df['age'], bins=8), observed=True)[target_col].agg(['mean', 'count'])
        logger.info("\n" + str(age_analysis))
    
    # Analyze Chest Pain vs Heart Disease
    if 'cp' in df.columns:
        logger.info("Chest Pain Type vs Heart Disease:")
        logger.info("Raw chest pain values: " + str(df['cp'].value_counts().head()))
        # Use actual values since this dataset uses string categories
        cp_analysis = df.groupby('cp')[target_col].agg(['mean', 'count'])
        logger.info("\n" + str(cp_analysis))
    
    # Analyze Cholesterol vs Heart Disease
    if 'chol' in df.columns:
        logger.info("Cholesterol vs Heart Disease:")
        # Exclude zero values (missing data)
        chol_data = df[df['chol'] > 0]
        if len(chol_data) > 0:
            chol_analysis = chol_data.groupby(pd.cut(chol_data['chol'], bins=5), observed=True)[target_col].agg(['mean', 'count'])
            logger.info("\n" + str(chol_analysis))
        logger.info(f"Zero cholesterol values (missing): {(df['chol'] == 0).sum()}")
    
    # Analyze Max Heart Rate vs Heart Disease
    if 'thalach' in df.columns:
        logger.info("Max Heart Rate vs Heart Disease:")
        hr_analysis = df.groupby(pd.cut(df['thalach'], bins=5), observed=True)[target_col].agg(['mean', 'count'])
        logger.info("\n" + str(hr_analysis))
    
    # Key clinical correlations
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if target_col in numeric_cols:
        logger.info("Top Correlations with Heart Disease:")
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        logger.info("\n" + str(correlations.head(10)))
    
    # Sample of key medical data
    key_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', target_col]
    available_cols = [col for col in key_cols if col in df.columns]
    logger.info("Sample of key medical data:\n" + str(df[available_cols].head(10)))

def main():
    """
    Main function to execute the complete Cleveland Heart Disease prediction pipeline
    """
    logger.info("CLEVELAND HEART DISEASE PREDICTION WITH LOGISTIC REGRESSION")
    
    # Step 1: Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        logger.error("Cannot proceed without dataset")
        return
    
    # Load the dataset
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded successfully: {df.shape}")
    except Exception as e:
        logger.exception(f"ERROR loading dataset: {e}")
        return
    
    # Step 2: Explore dataset
    df = explore_dataset(df)
    
    # Step 3: Preprocess data
    X, y = prepare_features_and_target(df)
    if X is None:
        logger.error("Data preparation failed")
        return
    
    # Step 4: Split and scale data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Train model
    pipeline, best_C = build_and_train_pipeline(X_train, y_train)
    
    # Step 6: Make predictions
    y_pred, y_pred_proba = make_predictions(pipeline, X_test)
    
    # Step 7: Evaluate model
    accuracy, precision, recall, f1, cm, extra_metrics = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Step 8: Analyze feature importance
    feature_importance = analyze_feature_importance(pipeline, X_train)
    
    # Create visualizations
    create_visualizations(cm, feature_importance, y_test=y_test, y_proba=y_pred_proba)

    # Save artifacts (model, feature importance, metrics)
    metrics_payload = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_C": best_C,
        **extra_metrics,
    }
    save_artifacts(pipeline, feature_importance, metrics_payload)
    
    # Final summary
    logger.info("=" * 60)
    logger.info("PROJECT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Model Accuracy: {accuracy:.1%}")
    logger.info(f"Dataset Size: {df.shape[0]} samples, {df.shape[1]} features")
    logger.info(f"Most Important Feature: {feature_importance.iloc[0]['feature']}")
    logger.info("Results visualized and saved to 'artifacts/'")

    # Analyze data quality
    analyze_data_quality(df)

if __name__ == "__main__":
    main()
