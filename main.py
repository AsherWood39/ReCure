import pandas as pd
import numpy as np
import joblib
import os
from data_handling import (
    prepare_data, parse_duration, SEVERITY_MAP, STRESS_MAP, 
    DOSHA_VAL_MAP, FEATURE_COLS, CAT_COLS,
    extract_clinical_features_nlp
)
from models import train_model

def get_counterfactual_ITE_recommendations(patient_features, model, stats, all_herbs, top_n=10):
    """
    Implements a High-Impact Causal Frame: Individual Treatment Effect (ITE).
    
    ITE = E[Y | Treatment, X] - E[Y | Baseline, X]
    Where Baseline = Average predicted outcome across all available herbs for this patient.
    
    Also implements Risk-Adjusted Standardized Utility:
    U = 1.0 * z_ite - 2.0 * z_risk
    """
    # Create batch of patient-herb pairs
    X_batch = pd.DataFrame([patient_features] * len(all_herbs))
    X_batch['herb_name'] = all_herbs
    
    # Interaction modeling (herb x disease)
    X_batch['herb_disease'] = X_batch['herb_name'] + "_" + str(patient_features.get('disease', 'unknown')).lower()
    
    # Predict Multi-Output [Improvement, Risk]
    preds = model.predict(X_batch[FEATURE_COLS])
    imp_raw = preds[:, 0]
    risk_raw = preds[:, 1]
    
    # --- Causal Frame: Estimate ITE ---
    # Baseline for this specific patient is the mean of all predicted outcomes
    patient_avg_outcome = imp_raw.mean()
    ite_scores = imp_raw - patient_avg_outcome
    
    # Z-Score Standardization for commemorative ranking
    # We use global population stats for commesurability
    z_ite = (ite_scores) / (stats['imp_sigma'] + 1e-9)
    z_risk = (risk_raw - stats['risk_mu']) / (stats['risk_sigma'] + 1e-9)
    
    # Risk-Adjusted Utility
    alpha = 1.0 # ITE weight
    beta = 2.0  # Risk penalty
    utility = (alpha * z_ite) - (beta * z_risk)
    
    results = pd.DataFrame({
        'herb': all_herbs,
        'improvement': imp_raw,
        'ite': ite_scores,
        'risk': risk_raw,
        'z_ite': z_ite,
        'z_risk': z_risk,
        'utility': utility
    })
    
    results['herb'] = results['herb'].str.title()
    results = results.sort_values(by='utility', ascending=False).head(top_n)
    
    return results, patient_avg_outcome

def rank_column_data_legacy(df, column_name, top_n=10):
    """Fallback similarity-based ranking for Diet/Yoga categories."""
    if column_name not in df.columns or df[column_name].dropna().empty:
        return [], []
    
    items_df = df.assign(item=df[column_name].str.split(',')).explode('item')
    items_df['item'] = items_df['item'].str.strip().str.lower()
    
    noisy_terms = ['none specific', 'unknown', 'placeholder', 'none', 'nil', 'tbd']
    items_df = items_df[~items_df['item'].str.contains('|'.join(noisy_terms), na=False)]
    items_df = items_df[items_df['item'] != '']
    
    rankings = items_df.groupby('item')['improvement_score'].agg(['mean', 'count']).sort_values(by=['mean', 'count'], ascending=False)
    
    top_items = [i.title() for i in rankings.head(top_n).index.tolist()]
    scores = rankings.head(top_n)['mean'].tolist()
    return top_items, scores

def predict_treatment_causal(patient_data, model, stats, lookup_data, all_herbs):
    """Orchestrates the personalized treatment effect estimation roadmap."""
    # 1. Feature Engineering
    df_p = pd.DataFrame([patient_data])
    df_p['symptom_severity_num'] = df_p['symptom_severity'].str.lower().map(SEVERITY_MAP).fillna(2)
    duration_days = parse_duration(df_p['duration_of_treatment'].iloc[0])
    df_p['duration_of_treatment_num'] = np.log1p(duration_days if not pd.isna(duration_days) else 7.0)
    df_p['stress_lvl_num'] = df_p['stress_levels'].str.lower().map(STRESS_MAP).fillna(2)
        
    p_dosha = df_p['prakriti_type'].str.lower().str.extract(r'(vata|pitta|kapha)')[0].map(DOSHA_VAL_MAP).fillna(0).iloc[0]
    d_dosha = df_p['doshas'].str.lower().str.extract(r'(vata|pitta|kapha)')[0].map(DOSHA_VAL_MAP).fillna(0).iloc[0]
    df_p['dosha_imbalance_score'] = np.abs(p_dosha - d_dosha)

    patient_baseline = df_p.iloc[0].to_dict()

    # 2. Causal ITE Ranking
    herb_recs, patient_baseline_pred = get_counterfactual_ITE_recommendations(patient_baseline, model, stats, all_herbs)

    # 3. Holistic Context
    disease = str(patient_data.get('disease', '')).lower().strip()
    primary_dosha = str(patient_data.get('doshas', 'vata')).lower()
    
    relevant_cases = pd.DataFrame()
    if 'doshas' in lookup_data.columns:
        relevant_cases = lookup_data[lookup_data['doshas'].str.lower().str.contains(primary_dosha, na=False)]
    else:
        relevant_cases = lookup_data
        
    ranking_source = relevant_cases
    if disease:
        disease_cases = relevant_cases[relevant_cases['disease'].str.lower().str.contains(disease, na=False)]
        if len(disease_cases) >= 3:
            ranking_source = disease_cases

    roadmap = {
        "herbs": herb_recs,
        "baseline_outcome": patient_baseline_pred,
        "diet_lifestyle": rank_column_data_legacy(ranking_source, 'diet_and_lifestyle_recommendations', 10),
        "yoga_therapy": rank_column_data_legacy(ranking_source, 'yoga_&_physical_therapy', 8),
        "prevention": rank_column_data_legacy(ranking_source, 'prevention', 8)
    }

    return roadmap

def main():
    print("\n========= ReCure: Personalized Risk-Adjusted Utility Framework =========")
    
    # 1. Pipeline Execution
    prepare_data()
    model = train_model()
    if model is None: return
    
    stats = joblib.load('model_info/model_stats.pkl')
    features_df = pd.read_csv('features.csv')
    all_herbs = sorted(features_df['herb_name'].dropna().unique().tolist())
    
    try:
        lookup_data = pd.read_csv("merged.csv")
    except:
        lookup_data = pd.DataFrame()

    # 2. Clinical Case Simulation
    clinical_note = (
        "Patient exhibits Pitta-centric constitution with high ambitious drive. "
        "Reports chronic acid reflux and occasional burning sensation in thorax. "
        "Symptoms worsen with spicy food. High workplace stress. "
        "Diagnosed with Amlapitta (Acid Gastritis)."
    )
    
    features = extract_clinical_features_nlp(clinical_note)
    if not features: return

    sample_patient = {
        'doshas': features.get('dosha', 'Pitta'),
        'prakriti_type': features.get('dosha', 'Pitta'),
        'disease': features.get('disease', 'Amlapitta'),
        'symptoms': features.get('symptoms', 'acid reflux, burning'),
        'symptom_severity': 'Moderate',
        'duration_of_treatment': '2-4 weeks',
        'stress_levels': features.get('stress_levels', 'High'),
        'sleep_patterns': 'Regular',
        'dietary_habits': 'Vegetarian',
        'body_size': features.get('body_size', 'Moderate'),
        'metabolism_type': 'Fast',
        'appetite': 'Strong'
    }
    
    res = predict_treatment_causal(sample_patient, model, stats, lookup_data, all_herbs)
    
    print(f"\n" + "="*85)
    print(f"RISK-ADJUSTED UTILITY RANKING | Individual Treatment Effect (ITE)")
    print("="*85)
    print(f"Estimated Patient Baseline Outcome: {res['baseline_outcome']:.2f}")
    print("-" * 85)
    print(f"{'#':<3} | {'Herb':<20} | {'ITE (z)':<10} | {'Risk (z)':<10} | {'Utility (U)'}")
    print("-" * 85)
    
    for i, row in enumerate(res['herbs'].itertuples(), 1):
        print(f"{i:<3} | {row.herb:<20} | {row.z_ite:<10.2f} | {row.z_risk:<10.2f} | {row.utility:.2f} (ITE: {row.ite:+.2f})")

    def print_section(title, data_tuple):
        items, scores = data_tuple
        if not items: return
        print(f"\n>>> {title}")
        print(f"{'#':<3} | {'Recommendation':<45} | {'Historical Score'}")
        print("-" * 75)
        for i, (item, score) in enumerate(zip(items, scores), 1):
            print(f"{i:<3} | {item[:45]:<45} | {float(score):.2f}")

    print_section("DIET & LIFESTYLE", res['diet_lifestyle'])
    print_section("YOGA & PHYSICAL THERAPY", res['yoga_therapy'])
    print_section("PREVENTION STRATEGIES", res['prevention'])

if __name__ == "__main__":
    main()
