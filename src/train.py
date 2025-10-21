# src/train.py
"""
Train RandomForest and XGBoost models to predict engagement metrics
Supports both scraped and API datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

# XGBoost is optional - only needed if model_type="xgboost"
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not available. Only Random Forest models will work.")

def train_scraped_model(
    filename="data/processed/scraped_processed.csv",
    target="views",
    model_type="random_forest",
    save_path="models/scraped_model.pkl"
):
    """
    Train model on scraped YouTube data
    Target: views (since we don't have likes/comments from scraping)
    """
    print("="*60)
    print("TRAINING MODEL ON SCRAPED DATA")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {filename}...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} videos")
    
    # Define features for scraped data
    numeric_features = [
        'duration_minutes',
        'time_since_upload_days',
        'title_len',
        'title_char_len',
        'title_upper_ratio',
        'views_per_day'
    ]
    
    boolean_features = [
        'has_channel',
        'is_music',
        'is_gaming',
        'is_educational'
    ]
    
    all_features = numeric_features + boolean_features
    
    # Check which features exist
    available_features = [f for f in all_features if f in df.columns]
    print(f"\nUsing {len(available_features)} features: {available_features}")
    
    # Prepare data
    df = df.dropna(subset=[target] + available_features)
    X = df[available_features].copy()
    y = df[target].copy()
    
    # Convert boolean to int
    for col in boolean_features:
        if col in X.columns:
            X[col] = X[col].astype(int)
    
    print(f"\nDataset size: {len(X)} samples")
    print(f"Target ({target}) stats:")
    print(f"  Mean: {y.mean():,.0f}")
    print(f"  Median: {y.median():,.0f}")
    print(f"  Min: {y.min():,.0f}")
    print(f"  Max: {y.max():,.0f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"\nTraining {model_type} model...")
    
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install it with: pip install xgboost")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluation
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)  # Compatible with all sklearn versions
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"\nTraining Set Performance:")
    print(f"  RMSE: {train_rmse:,.2f}")
    print(f"  MAE: {train_mae:,.2f}")
    print(f"  R²: {train_r2:.4f}")
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)  # Compatible with all sklearn versions
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:,.2f}")
    print(f"  MAE: {test_mae:,.2f}")
    print(f"  R²: {test_r2:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        importance_df = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_package = {
        'model': model,
        'scaler': scaler,
        'features': available_features,
        'target': target,
        'model_type': model_type,
        'metrics': {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    }
    joblib.dump(model_package, save_path)
    print(f"\nModel saved to {save_path}")
    
    return model_package, X_test, y_test, y_test_pred


def train_api_model(
    filename="data/processed/api_processed.csv",
    target="engagement_rate",
    model_type="random_forest",
    save_path="models/api_model.pkl"
):
    """
    Train model on API YouTube data
    Target: engagement_rate (or views)
    """
    print("="*60)
    print("TRAINING MODEL ON API DATA")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {filename}...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} videos")
    
    # Define features for API data
    numeric_features = [
        'duration_minutes',
        'time_since_upload_days',
        'title_len',
        'title_char_len',
        'title_upper_ratio',
        'desc_len',
        'desc_char_len',
        'num_tags',
        'views_per_day'
    ]
    
    # Add likes and comments if predicting views (not engagement_rate)
    if target == 'views':
        numeric_features.extend(['likes', 'comments'])
    
    # Category is categorical
    categorical_features = ['categoryId']
    
    all_features = numeric_features + categorical_features
    
    # Check which features exist
    available_numeric = [f for f in numeric_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    available_features = available_numeric + available_categorical
    
    print(f"\nUsing {len(available_features)} features:")
    print(f"  Numeric: {available_numeric}")
    print(f"  Categorical: {available_categorical}")
    
    # Prepare data
    df = df.dropna(subset=[target] + available_features)
    X = df[available_features].copy()
    y = df[target].copy()
    
    # One-hot encode categorical features
    if available_categorical:
        X = pd.get_dummies(X, columns=available_categorical, drop_first=True)
    
    final_features = list(X.columns)
    
    print(f"\nDataset size: {len(X)} samples")
    print(f"Final feature count (after encoding): {len(final_features)}")
    print(f"\nTarget ({target}) stats:")
    print(f"  Mean: {y.mean():.6f}")
    print(f"  Median: {y.median():.6f}")
    print(f"  Min: {y.min():.6f}")
    print(f"  Max: {y.max():.6f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"\nTraining {model_type} model...")
    
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install it with: pip install xgboost")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluation
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)  # Compatible with all sklearn versions
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"\nTraining Set Performance:")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  MAE: {train_mae:.6f}")
    print(f"  R²: {train_r2:.4f}")
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)  # Compatible with all sklearn versions
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  R²: {test_r2:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        importance_df = pd.DataFrame({
            'feature': final_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_package = {
        'model': model,
        'scaler': scaler,
        'features': final_features,
        'original_features': available_features,
        'target': target,
        'model_type': model_type,
        'metrics': {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    }
    joblib.dump(model_package, save_path)
    print(f"\nModel saved to {save_path}")
    
    return model_package, X_test, y_test, y_test_pred


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "scraped":
        train_scraped_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "api":
        train_api_model()
    else:
        print("Training both models...")
        print("\n")
        train_scraped_model()
        print("\n\n")
        train_api_model()
