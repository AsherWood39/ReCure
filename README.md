# ReCure: AI-Driven Ayurvedic Drug Repurposing

## Project Overview
ReCure is a prototype system that leverages machine learning to predict the effectiveness of personalized Ayurvedic herb recommendations. By integrating clinical patient data with traditional Ayurvedic Prakriti (constitution) traits, the system aims to provide a data-driven approach to drug repurposing and personalized medicine.

The core of the project is a **Random Forest Classifier** that analyzes factors such as:
- **Dosha Imbalance**: Quantifying the deviation between natural constitution and disease-induced state.
- **Symptom Severity**: Mapping qualitative symptom descriptions to numerical scales for machine analysis.
- **Herb Safety Index**: A data-driven score based on historical complication records to ensure safety-first recommendations.
- **Treatment Duration**: Standardized parsing of complex clinical duration formats into comparable numerical data.

## Key Features
- **Intelligent Data Integration**: Merges clinical datasets with Ayurvedic theoretical frameworks using advanced statistical aggregation (mode-based profiling).
- **Robust Feature Engineering**: Advanced text parsing for extracting meaningful numerical features from messy clinical data.
- **Safety-First Modeling**: Training logic optimized for Recall to ensure the system is cautious with herb effectiveness predictions.
- **Interpretability**: Automated generation of visualization reports to identify key clinical drivers in the decision-making process.

## Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

## Installation & Setup
1. **Navigate to the project folder** in your terminal.
2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Model
To train the model and generate the evaluation reports, execute the main classifier script:
```powershell
python Random_Forest_Model_Classifier.py
```

### Outputs
Upon successful execution, the script will:
- Display a **Classification Report** and **ROC-AUC Score** in the console.
- Generate `feature_importance.png`: A visual report of the top clinical factors influencing the model.
- Generate `confusion_matrix.png`: A breakdown of the model's performance on test data.

## Project Structure
- `Random_Forest_Model_Classifier.py`: Main execution script for data processing and model training.
- `AyurGenixAI_Dataset.csv`: Primary clinical dataset.
- `Prakriti_Tridosha_Dataset.csv`: Ayurvedic constitution reference data.
- `requirements.txt`: Python dependency list.
