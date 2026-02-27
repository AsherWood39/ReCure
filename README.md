# ReCure: Risk-Adjusted Utility Ranking Framework

## Project Overview
**ReCure** is a computational framework for Ayurvedic herb repurposing. It integrates traditional constitutional phenotyping (**Prakriti/Dosha**) and current symptom severity with a data-driven, causal inference engine. The system is designed as a specialized **Personalized Treatment Recommendation Engine** for traditional medicine research.

---

# Methodology: High-Impact Research Architecture
The framework is built on rigorous causal modeling to enable risk-aware treatment repurposing under simulated clinical conditions.

### 1. Data Integration and Causal Reformulation
The study integrates two heterogeneous datasets:
- **AyurGenixAI Clinical Dataset**: ~15,000 records linking diseases and symptoms to interventions.
- **Prakriti Tridosha Dataset**: ~1,200 records of physical and physiological traits.

**The Prakriti Bridge**: We implemented a "Holistic Bridge" merging strategy using 27 shared lifestyle and physiological features (stress levels, sleep patterns, dietary habits, appetite) to align clinical patient profiles with traditional Ayurvedic constitution. This correctly mimics the diagnostic process of an Ayurvedic Vaidya (doctor).

- **Causal Reformulation**: The dataset is structured into **Patient-Herb Treatment Episodes**, treating treatment ($H$) and baseline features ($X$) as independent predictors of clinical outcome ($Y$).

### 2. Modeling & Generalization (Rigor Phase)
To ensure methodological excellence, we implemented specialized validation controls:
- **GroupKFold (by Disease)**: The model is evaluated on its ability to generalize to **unseen diseases**, preventing interpolation bias and proving repurposing validity.
- **Noise Robustness**: Validated the stability of the recommendation logic by perturbing 20% of labels with Gaussian noise; the framework demonstrated high resilience (MAE delta < 0.15).
- **Multi-Model Benchmarking**: Compared 7 architectures (Linear, Lasso, Ridge, RF, GBR, XGBoost, SVR). **Random Forest** and **XGBoost** were selected for their superior capture of non-linear Herb $\times$ Patient interactions.

### 3. ITE Estimation & Ranking Framework
We formulate the repurposing task as an **Individual Treatment Effect (ITE)** estimation problem:
$$\tau(x, herb) = \hat{y}(x, herb) - \hat{y}(x, \text{baseline})$$
where the baseline is the mean predicted outcome for that specific patient profile across all available treatments.

The final **Risk-Adjusted Utility ($U$)** is calculated using standardized Z-scores:
$$U = \alpha \cdot z_{ite} - \beta \cdot z_{risk}$$
where $\alpha=1.0$ (Benefit weight) and $\beta=2.0$ (Safety penalty), enforcing a risk-averse prioritization strategy.

---

## Key Research Contributions
1. **Novel Merged Dataset**: First implementation of a longitudinal bridge between AyurGenixAI and Prakriti constitutional data for causal repurposing.
2. **Prakriti-Personalized Inference**: Unlike standard symptom-matching, ReCure models the non-linear interaction between a patient's biological phenotype and treatment response.
3. **Safety-First Ranking**: Prioritizes 90%+ recall on safety-critical risk prediction over raw accuracy, aligning with medical regulatory standards.
4. **Validating Ayurveda via AI**: Feature importance analysis consistently validates Ayurvedic tridosha theory, showing Dosha/Prakriti dominance in treatment success variance.

## Core Features
1. **Clinical Intake (NLP)**: Automated extraction of body traits, symptoms, and medical history from physician notes using **Groq (Llama 3.3)**.
2. **Appetite & Physiology Bridge**: Specialized domain mapping (e.g., mapping variable clinical appetites to traditional *Vishamagni/Tikshnagni*) for high-fidelity Prakriti alignment.
3. **Standardized ITE Estimation**: Individualized prediction of treatment gain relative to patient baseline, providing a true measure of herb-specific efficacy.
4. **Risk-Aware Decision Support**: Simultaneous prediction of clinical improvement and complication risk using multi-output regressors.
5. **Generalization across Unseen Diseases**: Verified cross-category performance via **GroupKFold** validation, ensuring zero leakage across diagnostic categories.
6. **Holistic Roadmap**: Provides ranked recommendations for **Herbs, Formulations, Diet, Yoga, and Prevention**.

## Workflow Pipeline
- **`data_handling.py`**: Handles NLP extraction and causal dataset restructuring.
- **`models.py`**: Multi-model arena with **GroupKFold** rigor and noise robustness testing.
- **`main.py`**: Orchestration of ITE estimation and risk-adjusted ranking.

## Installation & Setup
1. **Navigate to the project folder**.
2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **API Configuration**: Create a `.env` file and add your `GROQ_API_KEY`.

## Running the System
To execute the full data processing, rigorous training, and a personalized ITE diagnostic demo:
```powershell
python main.py
```

### Outputs & Artifacts (Stored in `model_info/`)
- **`features.csv`**: The interaction-aware clinical feature set.
- **`best_ayurvedic_model.pkl`**: The winner regression model.
- **`model_stats.pkl`**: Population statistics for standardized Z-score ranking.
- **`model_benchmarking.png`**: Visual reports of cross-validation and robustness.
