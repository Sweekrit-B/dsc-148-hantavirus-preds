# Hantavirus Mortality Prediction 
 
Binary classification models for predicting patient mortality from Hantavirus infection, combining clinical, epidemiological, environmental, and socioeconomic data. Models evaluated include Logistic Regression, Random Forest, XGBoost, and LightGBM.
 
---
 
## Project Structure
 
```
project/
├── notebooks
    ├──final_project.ipynb          
└── hantavirus_data/
    ├── hantavirus_clinical.csv          # Patient-level clinical records 
    └── hantavirus_country_yearly.csv    # Country-year epidemiological data
```
  
---
 
## Data Sources
 
| Source | What it provides | How it's accessed |
|---|---|---|
| [Kaggle — Hantavirus Andes Virus Global Epidemiology Dataset](https://www.kaggle.com/datasets/zkskhurram/hantavirus-andes-virus-global-epidemiology) | `hantavirus_clinical.csv` and `hantavirus_country_yearly.csv` | Manual download |
| [Open-Meteo Archive API](https://open-meteo.com/) | Average temperature and precipitation per country | `requests` (called automatically in notebook) |
| [World Bank — World Development Indicators](https://databank.worldbank.org/source/world-development-indicators) | Rural/urban population %, population density, GNI per capita, life expectancy | `wbgapi` (called automatically in notebook) |
| [REST Countries API](https://restcountries.com) | Latitude/longitude coordinates for ISO3 country codes | `requests` (called automatically in notebook) |
 
Download the two CSV files from Kaggle and place them in `hantavirus_data/` before running the notebook. The climate and socioeconomic data are fetched automatically at runtime, with no API keys required.
 
---
 
## Environment Setup
 
### Prerequisites
 
- Python **3.10**
- Conda or virtualenv
### 1. Create and activate the environment
 
```bash
conda create -n hantavirus-ml python=3.10
conda activate hantavirus-ml
```
 
### 2. Install dependencies
 
```bash
pip install pandas numpy matplotlib seaborn tqdm scikit-learn imbalanced-learn \
    xgboost lightgbm catboost shap textblob pgeocode pycountry wbgapi requests \
    mlxtend statsmodels MLstatkit jupyter
```
 
Or save the following to `requirements.txt` and run `pip install -r requirements.txt`:
 
```
pandas
numpy
matplotlib
seaborn
tqdm
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
shap
textblob
pgeocode
pycountry
wbgapi
requests
mlxtend
statsmodels
MLstatkit
jupyter
```
  
---
 
## Running the Notebook
 
```bash
jupyter notebook final_project.ipynb
```
 
Run cells **top to bottom**. The notebook is organized into the following sections:
 
### 1. Imports
All library imports. If any fail, revisit the install step above.
 
### 2. Create Dataset
- Loads `hantavirus_clinical.csv` and `hantavirus_country_yearly.csv`
- Merges the two tables on `country` and `year`
- Calls the Open-Meteo and World Bank APIs to fetch climate and socioeconomic features per country
- **This section makes live API requests** and includes a `time.sleep(10)` rate-limit delay per country, so it may take several minutes to run in full
### 3. EDA
Exploratory visualizations: outcome distribution, severity vs. mortality, viral load vs. mortality, symptom frequency, symptom-mortality breakdown, geographic variation, and a Pearson correlation heatmap.
 
### 4. Preprocessing
- Drops low-value columns (`who_region`, `iso_codes`, duplicate `syndrome`)
- Binarizes the comma-separated `symptoms` column via `MultiLabelBinarizer`
- Ordinal-encodes `severity` and `viral_load_category`
- One-hot encodes `country`, `syndrome`, and other nominal variables
- Creates a geographic availability indicator flag
- Sets `patient_id` as the DataFrame index
### 5. Modelling
 
#### Baseline
Logistic Regression trained on clinical features only (no country-year context), evaluated at threshold 0.5.
 
#### Initial Models (Stage 1)
All four models, Logistic Regression, Random Forest, XGBoost, LightGBM, trained on the full feature set with default hyperparameters and balanced class weights. ROC-AUC reported for initial comparison.
 
#### Tuned Models (Stage 2)
- **Hyperparameter tuning** via `GridSearchCV` with 5-fold cross-validation, optimizing Average Precision
- **Probability calibration** via `CalibratedClassifierCV` with Platt (sigmoid) scaling
- **Threshold optimization** by sweeping 0.01–1.00 and selecting the threshold that maximizes the combined F2 + MCC score
### 6. Statistical Analysis
- **Stratified bootstrap confidence intervals** (1000 iterations) for F2 and MCC across all models
- **McNemar's test** comparing each tuned model to the baseline
- **Cochran's Q test** comparing all models simultaneously
---
 
## Key Results
 
| Model | Threshold | F2 | MCC | F2 + MCC |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.50 | 0.2593 | 0.2927 | 0.5520 |
| Logistic Regression (Stage 2) | 0.51 | 0.5978 | 0.3992 | 0.9970 |
| Random Forest (Stage 2) | 0.20 | 0.5848 | 0.3919 | 0.9767 |
| XGBoost (Stage 2) | 0.11 | 0.5850 | 0.3941 | 0.9791 |
| **LightGBM (Stage 2)** | **0.12** | **0.5957** | **0.3961** | **0.9918** |
 
LightGBM was selected as the final model for its strong performance and statistically significant improvement over the baseline (McNemar's p = 2.01 × 10⁻¹²).
 
---
 
## Reproducing Results
 
Because the API data-fetching step is non-deterministic with respect to timing and minor API changes, exact numeric reproducibility requires saving the merged dataset after cell 8 and reloading it on subsequent runs. To do this, add the following after the data merge is complete:
 
```python
# Save merged dataset (run once after the API fetch completes)
data.to_csv("../hantavirus_data/hantavirus_merged.csv", index=False)
 
# Reload on subsequent runs instead of re-fetching
data = pd.read_csv("../hantavirus_data/hantavirus_merged.csv")
```
 
The train/test split does not use a fixed random seed in all locations — set `random_state=42` (or any fixed seed) consistently throughout the modelling section if exact split reproducibility is needed.
 
---
 
## Authors
 
Sweekrit Bhatnagar (`sbhatnagar@ucsd.edu`) and Sadhana Tadepalli (`satadepalli@ucsd.edu`)  
University of California, San Diego — DSC 148: Introduction to Data Mining
