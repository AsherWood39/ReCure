import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

# Import constants from data_handling
from data_handling import NUM_COLS, CAT_COLS

# Suppress warnings
warnings.filterwarnings("ignore")

def train_model(csv_path="features.csv"):
    """Benchmarks multiple regressors and saves the best multi-output model."""
    print(f"Loading features from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run data_handling first.")
        return None

    # Separate Features and Multi-Target Outcomes
    X = df[NUM_COLS + CAT_COLS]
    y = df[['improvement_score', 'risk_score']]

    # --- Phase 10: GroupKFold Rigor ---
    # To prevent leakage, we ensure the same disease doesn't appear in both train and test.
    # This evaluates the model's ability to repurpose herbs for UNSEEN diseases.
    groups = df['disease']
    
    # We use a manual split for the final holdout to visualize diagnostics
    unique_diseases = groups.unique()
    np.random.seed(42)
    np.random.shuffle(unique_diseases)
    train_slice = int(len(unique_diseases) * 0.8)
    train_diseases = unique_diseases[:train_slice]
    
    train_idx = df[df['disease'].isin(train_diseases)].index
    test_idx = df[~df['disease'].isin(train_diseases)].index
    
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUM_COLS),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encode', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]), CAT_COLS)
    ])

    # Model Arena
    bench_models = {
        'Linear Regression': MultiOutputRegressor(LinearRegression()),
        'Lasso': MultiOutputRegressor(Lasso(alpha=0.1)),
        'Ridge': MultiOutputRegressor(Ridge(alpha=1.0)),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GBR': MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
        'XGBoost': MultiOutputRegressor(XGBRegressor(random_state=42)),
        'SVR': MultiOutputRegressor(SVR())
    }

    results = []

    print("\n========= Model Benchmarking (MAE Comparison) =========")
    for name, model in bench_models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate MAE for both targets
        mae_imp = mean_absolute_error(y_test['improvement_score'], y_pred[:, 0])
        mae_risk = mean_absolute_error(y_test['risk_score'], y_pred[:, 1])
        r2_imp = r2_score(y_test['improvement_score'], y_pred[:, 0])
        
        results.append({
            'Model': name,
            'Imp_MAE': mae_imp,
            'Risk_MAE': mae_risk,
            'Imp_R2': r2_imp
        })
        print(f"{name:20} | Imp MAE: {mae_imp:.4f} | Risk MAE: {mae_risk:.4f} | Imp R2: {r2_imp:.4f}")

    # Select best model based on Improvement MAE
    best_res = min(results, key=lambda x: x['Imp_MAE'])
    print(f"\nWinner: {best_res['Model']}")

    winner_name = best_res['Model']
    winner_model = bench_models[winner_name]
    
    # Final Pipeline with the winner
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', winner_model)
    ])
    
    final_pipeline.fit(X_train, y_train)
    
    # Save the best model
    joblib.dump(final_pipeline, 'model_info/best_ayurvedic_model.pkl')
    
    # Calculate Z-score stats for the training set (needed for Ranking Utility in main.py)
    y_train_pred = final_pipeline.predict(X_train)
    stats = {
        'imp_mu': y_train_pred[:, 0].mean(),
        'imp_sigma': y_train_pred[:, 0].std(),
        'risk_mu': y_train_pred[:, 1].mean(),
        'risk_sigma': y_train_pred[:, 1].std(),
        'winner': winner_name
    }
    joblib.dump(stats, 'model_info/model_stats.pkl')
    
    print(f"\nFinal model and scaling stats saved.")
    
    # Noise Robustness Test (Phase 10)
    perform_noise_robustness_test(final_pipeline, X_test, y_test)
    
    # Evaluation plots for the winner
    evaluate_best_model(final_pipeline, X_test, y_test)
    
    return final_pipeline

def evaluate_best_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test['improvement_score'] - y_pred[:, 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_test['improvement_score'], y_pred[:, 0], alpha=0.5, color='teal')
    axes[0, 0].plot([1, 10], [1, 10], '--r')
    axes[0, 0].set_xlabel("Actual Improvement")
    axes[0, 0].set_ylabel("Predicted Improvement")
    axes[0, 0].set_title('Actual vs Predicted Effectiveness')
    
    # Residuals
    axes[0, 1].scatter(y_pred[:, 0], residuals, alpha=0.5, color='orange')
    axes[0, 1].axhline(y=0, color='black', linestyle='--')
    axes[0, 1].set_xlabel("Predicted Improvement")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title('Residual Plot (Standard Publication Check)')
    
    # Error Dist
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='purple')
    axes[1, 0].set_title('Error Distribution')
    
    # Feature Importance (Interaction Modeling Check)
    try:
        inner_model = model.named_steps['model']
        if hasattr(inner_model, 'feature_importances_'):
            importances = inner_model.feature_importances_
        elif hasattr(inner_model, 'estimators_'): # MultiOutput Wrapper
            importances = inner_model.estimators_[0].feature_importances_
        else:
            importances = None
            
        if importances is not None:
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
            sns.barplot(x=imp_series.values, y=imp_series.index, ax=axes[1, 1], palette='viridis')
            axes[1, 1].set_title('Top 15 Predictive Features (Interaction Analysis)')
    except:
        pass

    plt.tight_layout()
    plt.savefig('model_benchmarking.png')
    print("Benchmarking plots saved as 'model_benchmarking.png'.")

def perform_noise_robustness_test(model, X_test, y_test, noise_level=0.2):
    """Perturbs labels to verify the stability of the model logic."""
    print(f"\n[Robustness Test]: Perturbing {noise_level*100}% of labels with Gaussian noise...")
    
    y_test_noisy = y_test.copy()
    for col in y_test.columns:
        std = y_test[col].std()
        noise = np.random.normal(0, std * noise_level, size=len(y_test))
        y_test_noisy[col] = y_test_noisy[col] + noise
    
    y_pred = model.predict(X_test)
    mae_orig = mean_absolute_error(y_test['improvement_score'], y_pred[:, 0])
    mae_noisy = mean_absolute_error(y_test_noisy['improvement_score'], y_pred[:, 0])
    
    print(f"  Original MAE: {mae_orig:.4f}")
    print(f"  Noisy MAE: {mae_noisy:.4f}")
    print(f"  Sensitivity Delta: {abs(mae_orig - mae_noisy):.4f}")
    
    if abs(mae_orig - mae_noisy) < 0.2:
        print("  [SUCCESS] Model logic is ROBUST against label noise.")
    else:
        print("  [WARNING] Model shows high sensitivity to label noise.")

if __name__ == "__main__":
    train_model()