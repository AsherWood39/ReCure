import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def main():
    # 1. Load data
    try:
        ayur = pd.read_csv("AyurGenixAI_Dataset.csv")
        prakriti = pd.read_csv("Prakriti_Tridosha_Dataset.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Preprocess data
    # Rename for consistency
    ayur.columns = (
        ayur.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('/', '_')
    )
    ayur = ayur.rename(columns={'constitution_prakriti': 'prakriti_type'})

    prakriti.columns = (
        prakriti.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )

    # Normalize Lifestyle values for better matching
    def normalize_lifestyle(df, col):
        if col in df.columns:
            return df[col].astype(str).str.lower().str.strip()
        return None

    # Group prakriti dataset by dosha and aggregate traits using mode
    prakriti_summary = (
        prakriti
        .groupby('dosha')
        .agg({
            'body_size': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'complexion': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'metabolism_type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'climate_preference': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'stress_levels': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'sleep_patterns': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'dietary_habits': lambda x: x.mode().iloc[0] if not x.mode().empty else None
        })
        .reset_index()
        .rename(columns={'dosha': 'prakriti_type'})
    )

    # Merge with AyurGenixAI
    merged = pd.merge(
        ayur,
        prakriti_summary,
        on='prakriti_type',
        how='left',
        suffixes=('_ayur', '_prakriti')
    )

    # 3. Clinical Feature Engineering (Phase 3)
    # Define clinical target: treatment_effective
    # Logic: prognosis is 'good' AND complicaciones is 'none'
    merged['treatment_effective'] = (
        (merged['prognosis'].str.lower().str.contains('good', na=False)) & 
        (merged['complications'].str.lower().str.contains('none', na=False))
    ).astype(int)

    # Convert ordinal/numeric-like strings to numbers
    severity_map = {
        'mild': 1, 'mild to moderate': 2, 'moderate': 3,
        'moderate to high': 4, 'moderate to severe': 4,
        'high': 5, 'severe': 6
    }
    merged['symptom_severity_num'] = merged['symptom_severity'].str.lower().map(severity_map).fillna(2)

    # Log transform for duration_of_treatment (Phase 3)
    merged['duration_of_treatment_num'] = pd.to_numeric(merged['duration_of_treatment'], errors='coerce')
    merged['duration_of_treatment_num'] = np.log1p(merged['duration_of_treatment_num'].fillna(merged['duration_of_treatment_num'].median()))

    # Stress levels normalization (0-10 or ordinal)
    stress_map = {'low': 1, 'medium': 2, 'high': 3}
    merged['stress_lvl_num'] = merged['stress_levels_ayur'].str.lower().map(stress_map).fillna(2)

    # Dosha Imbalance Score Logic (Mental/Physical Deviation)
    # Mapping Doshas to simple scores for calculation
    dosha_val_map = {'vata': 1, 'pitta': 2, 'kapha': 3}
    merged['patient_dosha_val'] = merged['prakriti_type'].str.lower().str.extract(r'(vata|pitta|kapha)')[0].map(dosha_val_map).fillna(0)
    merged['disease_dosha_val'] = merged['doshas'].str.lower().str.extract(r'(vata|pitta|kapha)')[0].map(dosha_val_map).fillna(0)
    merged['dosha_imbalance_score'] = np.abs(merged['patient_dosha_val'] - merged['disease_dosha_val'])

    # Herb Safety Index (Phase 3)
    # Safety Index: 1 if complications="none" else 0.5
    merged['herb_safety_index'] = merged['complications'].str.lower().map(lambda x: 1.0 if 'none' in str(x) else 0.5).fillna(0.75)

    # Feature selection (Expanded based on Phase 2 & 3)
    num_cols = [
        'symptom_severity_num', 'stress_lvl_num', 
        'duration_of_treatment_num', 'dosha_imbalance_score',
        'herb_safety_index'
    ]
    cat_cols = [
        'doshas', 'prakriti_type', 'sleep_patterns_ayur',
        'dietary_habits_ayur', 'body_size', 'metabolism_type'
    ]
    feature_cols = num_cols + cat_cols

    X = merged[feature_cols]
    y = merged['treatment_effective']

    # 4. Model Pipeline with Safety Priority (Recall)
    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train Random Forest (Phase 4)
    # Prioritizing recall via class_weight
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=12, 
        random_state=42, 
        class_weight='balanced_subsample'
    )
    rf.fit(X_train, y_train)

    # 6. Evaluation (Phase 5: Safety-First)
    print("=== Training Results ===")
    print("Training Accuracy:", rf.score(X_train, y_train))
    print("Test Accuracy:", rf.score(X_test, y_test))

    # Detailed metrics
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    print("\n=== Safety-First Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
    
    # 5-fold CV focus on F1/Recall
    cv_f1 = cross_val_score(rf, X_processed, y, cv=5, scoring='f1_weighted')
    print(f"\n5-Fold CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    # Feature importance calculation
    importances = pd.DataFrame({
        'feature': preprocessor.get_feature_names_out(),
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # 6. Visualization
    # Top 10 Features
    top_features = importances.head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('Top 10 Features for Treatment Effectiveness')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature importance plot saved as 'feature_importance.png'.")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix plot saved as 'confusion_matrix.png'.")

if __name__ == "__main__":
    main()
