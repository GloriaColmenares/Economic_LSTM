# 📈 Economic LSTM, XGBoost & Statistical Models

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/Status-Research%20Project-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains implementations of:

- Deep Learning models (LSTM)
- Machine Learning models (XGBoost, two versions)
- Statistical models (ARMA, SARMA, ARMAX, SARMAX, Rolling Window variants, etc.)

for economic forecasting, along with post-processing and visualization tools used in the associated research.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure paths

Edit `paths.py` and update:

```python
BASE_INPUT_PATH = "your/input/path"
BASE_OUTPUT_PATH = "your/output/path"
```

---

## 🧠 Running the Models

Models can be run in two ways:

1. **From python** (`.py`)
2. **From the command line / SLURM**

---

## Option A — Run from Python

### Deep Learning Models

📁 `deep_learning_methods/`

- Open each python file  
- Click **Run**

⏱️ Estimated runtime (all models): **~230 minutes**

---

### Machine Learning Models (XGBoost)

📁 `machine_learning_methods/`

Available python files:

- `xgboost.py`
- `xgboost_v2.py`

To run:
- Open each python file
- Click **Run**

---

### Statistical Models

📁 `statistical_methods/`

- Open each python file  
- Click **Run**

⏱️ Estimated runtime (all models): **~242 minutes**

---

## Option B — Run from CLI / SLURM

Models can be executed directly as Python modules.

### Example

```bash
python3 -m machine_learning_methods.xgboost
```

---

## Running with SLURM

A sample submission script is provided:

📄 `submit-SLURM.sh`

### Example SLURM configuration

```bash
#!/bin/bash
#SBATCH --job-name=paper_model
#SBATCH --nodes=1
#SBATCH --partition=<your-partition>
#SBATCH --ntasks-per-node=32
#SBATCH --mem=512G
#SBATCH --error=job-%j-error.out
#SBATCH --output=job-%j-out.out
#SBATCH --export=ALL
#SBATCH --chdir=/path/to/project/01_Python_code
```

### Environment setup

```bash
source /path/to/venv/bin/activate
```

### Example submissions

#### Without exogenous features

```bash
sbatch --job-name=rw_sarma submit-SLURM.sh statistical_methods.rw_sarma
sbatch --job-name=xgboost submit-SLURM.sh machine_learning_methods.xgboost
```

#### With exogenous features

```bash
FEATURE_SET=1_4 sbatch --job-name=rwx_14ar submit-SLURM.sh statistical_methods.rw_armax
FEATURE_SET=4 sbatch --job-name=xgb4 submit-SLURM.sh machine_learning_methods.xgboost
```

### Notes

- The module to run is passed as an argument to `submit-SLURM.sh`
- `FEATURE_SET` is used for models with exogenous variables
- Logs are generated as:

```bash
job-%j-out.out
job-%j-error.out
```

---

## 📊 Post-Processing & Results

### Compute Metrics

📁 `post_processing/results_postprocessing.py`

Run to compute:

- Errors  
- Evaluation metrics (accuracy and economics) 

---

### Visualization

📁 `post_processing/results_visualisation.py`

Run all cells to generate plots used in the latest presentation.

---

## 🔄 Updating Market Data (Optional)

To fetch the latest market prices **before running models**:

📁 `data/data_creation.ipynb`

- Run all cells to:
  - Download publicly available data  
  - Build the dataset  

---

## ⚠️ Important Notes

### Execution Order

> ❗ All steps should be executed in order  
> Otherwise, errors may occur
> Selected exogenous parameters is not yet automated so if you download new data, exogenous parameters is still missing but you can run all models without exogenous parameters

---

### Data Limitations

- Exogenous data available **until February 2025 only**
- Models affected beyond this date:
  - ARMAX  
  - SARMAX  
  - LSTM (with exogenous features)  

---

### Runtime Considerations

Execution time depends on hardware.

**Reference system:**
- 48 CPUs @ 3.85 GHz  
- 256 GB RAM  
- 48 GB GPU memory  

---

## 🧩 Project Structure

```text
Economic_LSTM_XGboost/
├── 01_Python_code/
│   ├── data/
│   ├── deep_learning_methods/
│   ├── helper/
│   ├── machine_learning_methods/
│   │   ├── xgboost.ipynb
│   │   ├── xgboost.py
│   │   ├── xgboost_v2.ipynb
│   │   └── xgboost_v2.py
│   ├── post_processing/
│   ├── statistical_methods/
│   ├── __init__.py
│   ├── paths.py
│   ├── requirements.txt
│   ├── submit-SLURM.sh
│   └── READ_ME.txt
├── 02_Input_data/
│   ├── archive/
│   ├── aFRR_market_and_exogenous_factors_20190101_20250228.csv
│   ├── aFRR_market_and_exogenous_factors_20190101_20250228.xlsx
│   ├── processed_afrr_data.csv
│   └── selected_exogenous_factors_20210101_20250228.csv
├── 03_Output_paper/
│   ├── arma_model_output/
│   ├── armax_exog_1_3_model_output/
│   ├── armax_exog_1_4_model_output/
│   ├── armax_exog_1_model_output/
│   ├── armax_exog_4_model_output/
│   ├── bidirectional_lstm_model_output/
│   ├── lstm_features_1_3_model_output/
│   ├── lstm_features_1_4_model_output/
│   ├── lstm_features_1_model_output/
│   ├── lstm_features_4_model_output/
│   ├── lstm_model_output/
│   ├── postprocessed_data/
│   ├── rw_arma_3m_model_output/
│   ├── rw_arma_6m_model_output/
│   ├── rw_arma_12m_model_output/
│   ├── rw_armax_exog_1_3_3m_model_output/
│   ├── rw_armax_exog_1_3_6m_model_output/
│   ├── rw_armax_exog_1_3_12m_model_output/
│   └── rw_armax_exog_1_3m_model_output/
│   └── ...
```

---

## 🧩 Utility Modules

Used internally across the project:

- `data_processing.py`  
- `data_reading.py`  
- `helper_functions.py`  

✅ Automatically imported aFRR capacity prices
❌ No need to run manually  

---

## 📝 Paper Revision — 22.04.2026

### ✅ Updates

- Improved all models (DL, ML, statistical)  
- Added XGBoost implementations  
- Added SLURM logging  
- Ensured fair comparison across models  
- Added diagnostics  
- Improved post-processing  

---

### ⏳ Pending

- Diagnostics summary table  
- `sarmax_exog_1_4_model` output  

---

## 📦 Logs & Storage

Logs are excluded due to GitHub size limits.

Recommended:

```gitignore
04_Logs/
*.out
*.err
*.log
__pycache__/
```

---

## 📬 Notes

- Intended for research and reproducibility  
- Results may vary depending on environment  
- CLI / SLURM execution is recommended for large runs  