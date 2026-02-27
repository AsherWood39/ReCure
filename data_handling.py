import pandas as pd
import numpy as np  
import re
import warnings
import os
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
# Note: Ensure GROQ_API_KEY is set in your .env file
client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))

def extract_clinical_features_nlp(text):
    """
    Uses Groq API to extract standardized clinical features from physician notes,
    mapped to the schema of Prakriti_Tridosha_Dataset.csv.
    """
    if not text or pd.isna(text):
        return None
        
    system_prompt = (
        "You are an Ayurvedic Clinical Analyst. Analyze the patient description and extract the "
        "following features in a strict JSON format.\n\n"
        "1. Physical/Prakriti Traits (map to Prakriti schema): 'body_size', 'body_weight', 'bone_structure', 'complexion', 'appetite', 'sleep_patterns', 'stress_levels', 'dosha'.\n"
        "2. Clinical & Auxiliary Factors (map to AyurGenix schema): 'symptoms', 'disease', 'medical_history', 'family_history', 'allergies'.\n\n"
        "Rules:\n"
        "- 'dosha' should be 'Vata', 'Pitta', 'Kapha', or combinations (e.g., 'Vata+Pitta').\n"
        "- 'appetite' should be mapped to: 'Strong, Unbearable', 'Slow but steady', or 'Irregular, Scanty'.\n"
        "- 'symptoms' should be a comma-separated string of identified physical symptoms.\n"
        "- 'disease' should be the name of the identified condition (if mentioned).\n"
        "- Use clear summary strings for 'medical_history', 'family_history', and 'allergies'.\n"
        "- If an attribute is missing, use null.\n"
        "Example output: {'body_size': 'Slim', 'appetite': 'Irregular, Scanty', 'dosha': 'Vata', 'symptoms': 'headache, fever', 'disease': 'Vishamajvara', 'medical_history': 'Asthma', 'family_history': 'Diabetes', 'allergies': 'Pollen'}"
    )
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        import json
        result = json.loads(completion.choices[0].message.content)
        # Normalize keys to match dataset if needed
        return result
    except Exception as e:
        print(f"NLP Extraction Error: {e}")
        return None

# Suppress warnings
warnings.filterwarnings("ignore")

# Static Maps for consistency across project
SEVERITY_MAP = {
    'mild': 1, 'mild to moderate': 2, 'moderate': 3,
    'moderate to high': 4, 'moderate to severe': 4,
    'high': 5, 'severe': 6
}

STRESS_MAP = {'low': 1, 'moderate': 2, 'high': 3}

DOSHA_VAL_MAP = {'vata': 1, 'pitta': 2, 'kapha': 3}

# Redesigned Features (Phase 9: Interaction Modeling)
NUM_COLS = [
    'symptom_severity_num', 'stress_lvl_num', 
    'duration_of_treatment_num', 'dosha_imbalance_score'
]

CAT_COLS = [
    'doshas', 'prakriti_type', 'sleep_patterns',
    'dietary_habits', 'body_size', 'metabolism_type', 'appetite',
    'herb_name', 'disease', 'herb_disease'
]

FEATURE_COLS = NUM_COLS + CAT_COLS

def calculate_safety_index(complications_text):
    """Calculates a granular safety index (0.1 to 1.0) based on severity."""
    text = str(complications_text).lower()
    
    if any(x in text for x in ['none', 'no complications', 'no side effects', 'nil']):
        return 1.0
    if any(x in text for x in ['mild', 'headache', 'minor', 'occasional']):
        return 0.8
    if any(x in text for x in ['moderate', 'irritation', 'rash', 'fatigue', 'nausea']):
        return 0.6
    if any(x in text for x in ['severe', 'hypertension', 'vomiting', 'bleeding', 'inflammation']):
        return 0.4
    if any(x in text for x in ['failure', 'cancer', 'stroke', 'death', 'permanent', 'chronic']):
        return 0.2
    return 0.5

def parse_duration(duration_str):
    """Parses duration strings into numerical days."""
    if pd.isna(duration_str):
        return np.nan
    duration_str = str(duration_str).lower()
    if 'lifelong' in duration_str or 'lifetime' in duration_str:
        return 3650.0  # ~10 years
    if 'variable' in duration_str:
        return np.nan
    
    nums = re.findall(r'\d+\.?\d*', duration_str)
    if not nums:
        return np.nan
    
    val = np.mean([float(n) for n in nums])
    if 'week' in duration_str:
        val *= 7
    elif 'month' in duration_str:
        val *= 30
    elif 'year' in duration_str:
        val *= 365
    return val

def label_targets(row):
    """
    Separate outcomes into Effectiveness (Y) and Risk (R).
    Strictly leakage-free: 
    - Effectiveness depends ONLY on prognosis.
    - Risk depends ONLY on complications.
    """
    prog = str(row['prognosis']).lower()
    comp = str(row['complications']).lower()
    
    # --- Y: Improvement Score (1-10) ---
    y_score = 5 # Default neutral
    if any(x in prog for x in ['recovery', 'complete relief', 'cured']): y_score += 5
    elif any(x in prog for x in ['good', 'corrected', 'relief', 'improvement']): y_score += 3
    elif 'managed' in prog: y_score += 1
    if any(x in prog for x in ['progressive', 'failure', 'death']): y_score -= 4
    elif any(x in prog for x in ['chronic', 'lifelong', 'damage']): y_score -= 2
    y_score = max(1, min(10, y_score))

    # --- R: Risk Score (0-1) ---
    r_score = 0.0 # Safe by default
    if any(x in comp for x in ['failure', 'cancer', 'stroke', 'death']): r_score = 1.0
    elif any(x in comp for x in ['severe', 'permanent']): r_score = 0.8
    elif any(x in comp for x in ['moderate', 'irritation', 'rash', 'fatigue', 'nausea']): r_score = 0.5
    elif any(x in comp for x in ['mild', 'headache', 'minor']): r_score = 0.2
    
    return pd.Series([y_score, r_score], index=['improvement_score', 'risk_score'])

def prepare_data(data_path="datasets"):
    """Loads, preprocesses, and saves features for model training."""
    print("Starting data preparation...")
    
    try:
        ayur = pd.read_csv(os.path.join(data_path, "AyurGenixAI_Dataset.csv"))
        prakriti = pd.read_csv(os.path.join(data_path, "Prakriti_Tridosha_Dataset.csv"))
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return None

    # Standardize column names
    ayur.columns = ayur.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    ayur = ayur.rename(columns={'constitution_prakriti': 'prakriti_type'})
    
    # Aggressive Dosha normalization: keep only vata/pitta/kapha and join with +
    def normalize_dosha(text):
        if pd.isna(text): return ""
        found = re.findall(r'(vata|pitta|kapha)', str(text).lower())
        return "+".join(sorted(list(set(found))))

    for col in ['prakriti_type', 'doshas']:
        ayur[col] = ayur[col].apply(normalize_dosha)

    prakriti.columns = prakriti.columns.str.strip().str.lower().str.replace(' ', '_')
    prakriti['dosha'] = prakriti['dosha'].apply(normalize_dosha)

    # --- Bridge Maps for Holistic Alignment ---
    SLEEP_BRIDGE = {
        'irregular': 'short', 'poor': 'short', 'disrupted': 'short', 
        'fatigue': 'short', 'regular': 'moderate'
    }
    STRESS_BRIDGE = {
        'very high': 'high', 'high': 'high', 'moderate': 'moderate', 'low': 'low'
    }
    ACTIVITY_BRIDGE = {
        'low': 'sedentary', 'moderate': 'moderate', 'high': 'high'
    }
    DIET_BRIDGE = {
        'vegetarian': 'vegetarian', 'non-vegetarian': 'omnivorous', 
        'vegan': 'vegan', 'balanced': 'omnivorous'
    }
    # Map the 3 existing types to standardized, single-word labels
    APPETITE_BRIDGE = {
        'slow but steady': 'slow',      # Kapha (Mandagni)
        'strong, unbearable': 'sharp',  # Pitta (Tikshnagni)
        'irregular, scanty': 'variable' # Vata (Vishamagni)
    }

    # Normalize clinical data for a robust merge
    # Access original names from AyurGenixAI_Dataset.csv (renamed to lower_case_with_underscores)
    ayur['sleep_patterns'] = ayur['sleep_patterns'].str.lower().apply(lambda x: next((v for k, v in SLEEP_BRIDGE.items() if k in str(x)), 'moderate'))
    ayur['stress_levels'] = ayur['stress_levels'].str.lower().apply(lambda x: next((v for k, v in STRESS_BRIDGE.items() if k in str(x)), 'moderate'))
    
    # physical_activity_levels has an 's' in the ayur dataset
    ayur['physical_activity_level'] = ayur['physical_activity_levels'].str.lower().apply(lambda x: next((v for k, v in ACTIVITY_BRIDGE.items() if k in str(x)), 'moderate'))
    ayur['dietary_habits'] = ayur['dietary_habits'].str.lower().apply(lambda x: next((v for k, v in DIET_BRIDGE.items() if k in str(x)), 'omnivorous'))

    prakriti['appetite'] = prakriti['appetite'].str.lower().apply(lambda x: next((v for k, v in APPETITE_BRIDGE.items() if k in str(x)), 'Slow but steady'))

    # Summarize Prakriti dataset with lifestyle keys
    merge_keys = ['prakriti_type', 'sleep_patterns', 'stress_levels', 'dietary_habits', 'physical_activity_level']
    
    prakriti_summary = prakriti.rename(columns={'dosha': 'prakriti_type'}).groupby(merge_keys).agg({
        'body_size': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'complexion': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'metabolism_type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'climate_preference': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'appetite': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'digestion_quality': lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).reset_index()

    # --- 1. Explode Dataset (Patient-Herb Pairs) ---
    ayur = ayur.assign(herb_name=ayur['ayurvedic_herbs'].str.split(',')).explode('herb_name')
    ayur['herb_name'] = ayur['herb_name'].str.strip().str.lower()
    ayur = ayur[~ayur['herb_name'].str.contains('none specific|unknown|placeholder|none|nil', na=False)]
    ayur = ayur[ayur['herb_name'] != '']

    # --- 2. Label Targets (Leakage-Free) ---
    targets = ayur.apply(label_targets, axis=1)
    ayur = pd.concat([ayur, targets], axis=1)

    # --- 3. Feature Engineering ---
    ayur['symptom_severity_num'] = ayur['symptom_severity'].str.lower().map(SEVERITY_MAP).fillna(2)
    ayur['duration_of_treatment_num'] = ayur['duration_of_treatment'].apply(parse_duration)
    median_duration = ayur['duration_of_treatment_num'].median()
    ayur['duration_of_treatment_num'] = np.log1p(ayur['duration_of_treatment_num'].fillna(median_duration if not pd.isna(median_duration) else 7.0))
    ayur['stress_lvl_num'] = ayur['stress_levels'].str.lower().map(STRESS_MAP).fillna(2)

    def get_average_dosha_val(dosha_string):
        found = re.findall(r'(vata|pitta|kapha)', str(dosha_string).lower())
        if not found: return 0
        values = [DOSHA_VAL_MAP.get(d, 0) for d in found]
        return sum(values) / len(values)

    ayur['patient_dosha_val'] = ayur['prakriti_type'].apply(get_average_dosha_val)
    ayur['disease_dosha_val'] = ayur['doshas'].apply(get_average_dosha_val)
    ayur['dosha_imbalance_score'] = np.abs(ayur['patient_dosha_val'] - ayur['disease_dosha_val'])

    # --- 4. Causal Interaction Modeling ---
    ayur['herb_disease'] = ayur['herb_name'] + "_" + ayur['disease'].astype(str)

    # --- 5. Holistic Merging ---
    merge_keys = ['prakriti_type', 'sleep_patterns', 'stress_levels', 'dietary_habits', 'physical_activity_level']
    prakriti_summary = prakriti.rename(columns={'dosha': 'prakriti_type'}).groupby(merge_keys).agg({
        'body_size': lambda x: x.mode().iloc[0] if not x.mode().empty else 'moderate',
        'metabolism_type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'balanced',
        'appetite': lambda x: x.mode().iloc[0] if not x.mode().empty else 'slow'
    }).reset_index()

    merged = pd.merge(ayur, prakriti_summary, on=merge_keys, how='left')

    # Normalize categorical columns
    for col in CAT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.lower().str.strip()

    merged.to_csv("merged.csv", index=False)

    # Save to features.csv
    output_cols = FEATURE_COLS + ['improvement_score', 'risk_score']
    features_df = merged[output_cols].dropna(subset=['improvement_score'])
    features_df.to_csv("features.csv", index=False)
    
    print(f"Data preparation complete. Features saved to 'features.csv'. Shape: {features_df.shape}")
    return features_df

if __name__ == "__main__":
    prepare_data()