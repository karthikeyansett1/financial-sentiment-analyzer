"""
model.py — Train and evaluate ML models.

WHAT WE'RE PREDICTING:
Binary classification: will the stock go UP (1) or DOWN (0) tomorrow?

WHY THESE THREE MODELS:
1. Logistic Regression — simple baseline, interpretable, interviewers expect it
2. Random Forest — ensemble of decision trees, handles non-linear relationships
3. XGBoost — gradient boosting, usually the best performer on tabular data

KEY CONCEPT - TIME-BASED SPLIT:
We do NOT use random train/test split. Why? Because in finance, using
future data to predict the past is cheating (called "data leakage").
We train on earlier dates and test on later dates — just like real life.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.database import load_from_db


# Features to use for modeling (exclude identifiers and target-related columns)
FEATURE_COLS = [
    # Sentiment features
    "avg_positive", "avg_negative", "avg_neutral", "sentiment_std",
    "news_count", "pos_ratio", "neg_ratio", "sentiment_spread",
    # Technical features
    "daily_return", "price_vs_ma5", "price_vs_ma10",
    "volatility_5d", "volume_ratio", "momentum_3d",
]

TARGET_COL = "target"


def prepare_data():
    """
    Load features and create time-based train/test split.
    
    WHY TIME-BASED SPLIT:
    If we randomly split, training data might include April 1st 
    while testing on March 28th — we'd be using the future to 
    predict the past. Time-based split prevents this.
    """
    df = load_from_db("SELECT * FROM features")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Drop any rows with NaN in features
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    
    # Time-based split: first 75% for training, last 25% for testing
    split_idx = int(len(df) * 0.75)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL].astype(int)
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL].astype(int)
    
    # Scale features — important for Logistic Regression
    # Tree-based models (RF, XGBoost) don't need scaling but it doesn't hurt
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                   columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                  columns=FEATURE_COLS, index=X_test.index)
    
    print(f"Training set: {len(X_train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test set:     {len(X_test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"Features:     {len(FEATURE_COLS)}")
    print(f"Train target: {dict(y_train.value_counts())}")
    print(f"Test target:  {dict(y_test.value_counts())}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate():
    """
    Train three models, compare performance, save the best one.
    """
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "XGBoost": None,  # Will import separately
    }
    
    # Import XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="logloss"
        )
    except ImportError:
        print("⚠️  XGBoost not installed, skipping. Run: pip install xgboost")
    
    results = {}
    best_model = None
    best_score = 0
    best_name = ""
    
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        if model is None:
            continue
            
        print(f"\n--- {name} ---")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.0
        
        results[name] = {
            "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "AUC": auc
        }
        
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print(f"AUC:       {auc:.3f}")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\n{classification_report(y_test, y_pred, target_names=['DOWN', 'UP'])}")
        
        # Track best model by F1 score
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
    
    # Summary table
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    results_df = pd.DataFrame(results).T
    print(results_df.round(3))
    
    # Save best model
    print(f"\n🏆 Best model: {best_name} (F1: {best_score:.3f})")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Saved best model and scaler to models/")
    
    return best_model, results_df


if __name__ == "__main__":
    model, results = train_and_evaluate()