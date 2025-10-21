# src/evaluate.py
"""
Evaluate trained models and generate comprehensive visualizations
Compares scraped vs API model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Seaborn is optional - only for styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("WARNING: Seaborn not available. Using default matplotlib styling.")
    plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (10, 6)

def load_model(model_path):
    """Load a saved model package"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """Create actual vs predicted scatter plot"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()

def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    """Create residual plot"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, s=50)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()

def plot_feature_importance(model_package, top_n=15, title="Feature Importance", save_path=None):
    """Plot feature importance"""
    model = model_package['model']
    
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_ attribute")
        return
    
    features = model_package['features']
    importances = model.feature_importances_
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'], align='center')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()
    
    return importance_df

def compare_models(scraped_model_path="models/scraped_model.pkl", 
                  api_model_path="models/api_model.pkl",
                  output_dir="reports/figures"):
    """
    Compare performance between scraped and API models
    """
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
    scraped_pkg = load_model(scraped_model_path)
    api_pkg = load_model(api_model_path)
    
    # Extract metrics
    scraped_metrics = scraped_pkg['metrics']
    api_metrics = api_pkg['metrics']
    
    # Print comparison
    print("\n" + "="*60)
    print("SCRAPED DATA MODEL")
    print("="*60)
    print(f"Target: {scraped_pkg['target']}")
    print(f"Model Type: {scraped_pkg['model_type']}")
    print(f"Features: {len(scraped_pkg['features'])}")
    print(f"\nTraining Performance:")
    print(f"  RMSE: {scraped_metrics['train_rmse']:,.2f}")
    print(f"  MAE: {scraped_metrics['train_mae']:,.2f}")
    print(f"  R²: {scraped_metrics['train_r2']:.4f}")
    print(f"\nTest Performance:")
    print(f"  RMSE: {scraped_metrics['test_rmse']:,.2f}")
    print(f"  MAE: {scraped_metrics['test_mae']:,.2f}")
    print(f"  R²: {scraped_metrics['test_r2']:.4f}")
    
    print("\n" + "="*60)
    print("API DATA MODEL")
    print("="*60)
    print(f"Target: {api_pkg['target']}")
    print(f"Model Type: {api_pkg['model_type']}")
    print(f"Features: {len(api_pkg['features'])}")
    print(f"\nTraining Performance:")
    print(f"  RMSE: {api_metrics['train_rmse']:.6f}")
    print(f"  MAE: {api_metrics['train_mae']:.6f}")
    print(f"  R²: {api_metrics['train_r2']:.4f}")
    print(f"\nTest Performance:")
    print(f"  RMSE: {api_metrics['test_rmse']:.6f}")
    print(f"  MAE: {api_metrics['test_mae']:.6f}")
    print(f"  R²: {api_metrics['test_r2']:.4f}")
    
    # Create comparison visualization
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Model comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['Scraped\nData', 'API\nData']
    
    # R² comparison
    r2_scores = [scraped_metrics['test_r2'], api_metrics['test_r2']]
    axes[0].bar(models, r2_scores, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, max(r2_scores) * 1.2])
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # MAE comparison (normalized)
    mae_scraped_norm = scraped_metrics['test_mae'] / scraped_metrics['test_mae']
    mae_api_norm = api_metrics['test_mae'] / api_metrics['test_mae']
    axes[1].bar(models, [mae_scraped_norm, mae_api_norm], color=['#3498db', '#e74c3c'])
    axes[1].set_ylabel('Normalized MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error (Normalized)', fontsize=12, fontweight='bold')
    axes[1].text(0, mae_scraped_norm + 0.01, f'{scraped_metrics["test_mae"]:,.2f}', ha='center', va='bottom')
    axes[1].text(1, mae_api_norm + 0.01, f'{api_metrics["test_mae"]:.6f}', ha='center', va='bottom')
    
    # Feature count comparison
    feature_counts = [len(scraped_pkg['features']), len(api_pkg['features'])]
    axes[2].bar(models, feature_counts, color=['#3498db', '#e74c3c'])
    axes[2].set_ylabel('Number of Features', fontsize=12)
    axes[2].set_title('Feature Count Comparison', fontsize=12, fontweight='bold')
    for i, v in enumerate(feature_counts):
        axes[2].text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = f"{output_dir}/model_comparison.png"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison to {save_path}")
    plt.close()
    
    return scraped_pkg, api_pkg

def analyze_engagement_trends(api_data_path="data/processed/api_processed.csv", 
                              output_dir="reports/figures"):
    """
    Analyze engagement trends by category, duration, and upload time
    """
    print("\n" + "="*60)
    print("ENGAGEMENT TRENDS ANALYSIS")
    print("="*60)
    
    df = pd.read_csv(api_data_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Engagement by Category
    ax1 = plt.subplot(2, 2, 1)
    if 'categoryId' in df.columns and 'engagement_rate' in df.columns:
        category_engagement = df.groupby('categoryId')['engagement_rate'].agg(['mean', 'count'])
        category_engagement = category_engagement[category_engagement['count'] >= 5].sort_values('mean', ascending=False).head(10)
        
        ax1.barh(range(len(category_engagement)), category_engagement['mean'])
        ax1.set_yticks(range(len(category_engagement)))
        ax1.set_yticklabels([f"Cat {cat}" for cat in category_engagement.index])
        ax1.set_xlabel('Mean Engagement Rate')
        ax1.set_title('Engagement Rate by Category (Top 10)', fontweight='bold')
        ax1.invert_yaxis()
    
    # 2. Engagement by Video Length
    ax2 = plt.subplot(2, 2, 2)
    if 'duration_minutes' in df.columns and 'engagement_rate' in df.columns:
        # Create duration bins
        df['duration_category'] = pd.cut(df['duration_minutes'], 
                                         bins=[0, 5, 10, 20, 60, float('inf')],
                                         labels=['0-5 min', '5-10 min', '10-20 min', '20-60 min', '60+ min'])
        duration_engagement = df.groupby('duration_category', observed=True)['engagement_rate'].mean()
        
        ax2.bar(range(len(duration_engagement)), duration_engagement.values)
        ax2.set_xticks(range(len(duration_engagement)))
        ax2.set_xticklabels(duration_engagement.index, rotation=45)
        ax2.set_ylabel('Mean Engagement Rate')
        ax2.set_title('Engagement Rate by Video Duration', fontweight='bold')
    
    # 3. Views by Category
    ax3 = plt.subplot(2, 2, 3)
    if 'categoryId' in df.columns and 'views' in df.columns:
        category_views = df.groupby('categoryId')['views'].agg(['mean', 'count'])
        category_views = category_views[category_views['count'] >= 5].sort_values('mean', ascending=False).head(10)
        
        ax3.barh(range(len(category_views)), category_views['mean'])
        ax3.set_yticks(range(len(category_views)))
        ax3.set_yticklabels([f"Cat {cat}" for cat in category_views.index])
        ax3.set_xlabel('Mean Views')
        ax3.set_title('Average Views by Category (Top 10)', fontweight='bold')
        ax3.invert_yaxis()
    
    # 4. Engagement vs Time Since Upload
    ax4 = plt.subplot(2, 2, 4)
    if 'time_since_upload_days' in df.columns and 'engagement_rate' in df.columns:
        # Create time bins
        df['age_category'] = pd.cut(df['time_since_upload_days'],
                                    bins=[0, 30, 90, 180, 365, float('inf')],
                                    labels=['< 1 month', '1-3 months', '3-6 months', '6-12 months', '1+ year'])
        age_engagement = df.groupby('age_category', observed=True)['engagement_rate'].mean()
        
        ax4.bar(range(len(age_engagement)), age_engagement.values)
        ax4.set_xticks(range(len(age_engagement)))
        ax4.set_xticklabels(age_engagement.index, rotation=45)
        ax4.set_ylabel('Mean Engagement Rate')
        ax4.set_title('Engagement Rate by Video Age', fontweight='bold')
    
    plt.tight_layout()
    save_path = f"{output_dir}/engagement_trends.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved engagement trends to {save_path}")
    plt.close()

def generate_full_report(scraped_model_path="models/scraped_model.pkl",
                        api_model_path="models/api_model.pkl",
                        scraped_data_path="data/processed/scraped_processed.csv",
                        api_data_path="data/processed/api_processed.csv",
                        output_dir="reports/figures"):
    """
    Generate complete evaluation report with all visualizations
    """
    print("\n" + "="*80)
    print(" "*20 + "FULL EVALUATION REPORT")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load models and compare
    print("\n[1/6] Loading and comparing models...")
    scraped_pkg, api_pkg = compare_models(scraped_model_path, api_model_path, output_dir)
    
    # 2. Reload data and make predictions for visualization
    print("\n[2/6] Generating predictions for visualizations...")
    
    # Scraped model visualizations
    scraped_df = pd.read_csv(scraped_data_path)
    scraped_features = scraped_pkg['features']
    scraped_target = scraped_pkg['target']
    
    scraped_df_clean = scraped_df.dropna(subset=[scraped_target] + scraped_features)
    X_scraped = scraped_df_clean[scraped_features].copy()
    
    # Convert boolean columns
    for col in X_scraped.columns:
        if X_scraped[col].dtype == bool:
            X_scraped[col] = X_scraped[col].astype(int)
    
    y_scraped_true = scraped_df_clean[scraped_target]
    X_scraped_scaled = scraped_pkg['scaler'].transform(X_scraped)
    y_scraped_pred = scraped_pkg['model'].predict(X_scraped_scaled)
    
    # API model visualizations
    api_df = pd.read_csv(api_data_path)
    api_features = api_pkg['original_features']
    api_target = api_pkg['target']
    
    api_df_clean = api_df.dropna(subset=[api_target] + api_features)
    X_api = api_df_clean[api_features].copy()
    
    # One-hot encode if needed
    if 'categoryId' in X_api.columns:
        X_api = pd.get_dummies(X_api, columns=['categoryId'], drop_first=True)
    
    # Ensure all features from training are present
    for feat in api_pkg['features']:
        if feat not in X_api.columns:
            X_api[feat] = 0
    
    X_api = X_api[api_pkg['features']]
    y_api_true = api_df_clean[api_target]
    X_api_scaled = api_pkg['scaler'].transform(X_api)
    y_api_pred = api_pkg['model'].predict(X_api_scaled)
    
    # 3. Create actual vs predicted plots
    print("\n[3/6] Creating actual vs predicted plots...")
    plot_actual_vs_predicted(y_scraped_true, y_scraped_pred, 
                            title=f"Scraped Model: Actual vs Predicted {scraped_target}",
                            save_path=f"{output_dir}/scraped_actual_vs_predicted.png")
    
    plot_actual_vs_predicted(y_api_true, y_api_pred,
                            title=f"API Model: Actual vs Predicted {api_target}",
                            save_path=f"{output_dir}/api_actual_vs_predicted.png")
    
    # 4. Create residual plots
    print("\n[4/6] Creating residual plots...")
    plot_residuals(y_scraped_true, y_scraped_pred,
                  title=f"Scraped Model: Residual Plot",
                  save_path=f"{output_dir}/scraped_residuals.png")
    
    plot_residuals(y_api_true, y_api_pred,
                  title=f"API Model: Residual Plot",
                  save_path=f"{output_dir}/api_residuals.png")
    
    # 5. Create feature importance plots
    print("\n[5/6] Creating feature importance plots...")
    scraped_importance = plot_feature_importance(scraped_pkg, top_n=15,
                                                title="Scraped Model: Top 15 Important Features",
                                                save_path=f"{output_dir}/scraped_feature_importance.png")
    
    api_importance = plot_feature_importance(api_pkg, top_n=15,
                                            title="API Model: Top 15 Important Features",
                                            save_path=f"{output_dir}/api_feature_importance.png")
    
    # 6. Analyze engagement trends
    print("\n[6/6] Analyzing engagement trends...")
    analyze_engagement_trends(api_data_path, output_dir)
    
    print("\n" + "="*80)
    print("FULL REPORT GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - model_comparison.png")
    print("  - scraped_actual_vs_predicted.png")
    print("  - api_actual_vs_predicted.png")
    print("  - scraped_residuals.png")
    print("  - api_residuals.png")
    print("  - scraped_feature_importance.png")
    print("  - api_feature_importance.png")
    print("  - engagement_trends.png")
    
    return {
        'scraped_metrics': scraped_pkg['metrics'],
        'api_metrics': api_pkg['metrics'],
        'scraped_importance': scraped_importance,
        'api_importance': api_importance
    }

if __name__ == "__main__":
    generate_full_report()
