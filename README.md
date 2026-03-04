# Transaction Risk Intelligence Platform

End-to-end pipeline for detecting fraudulent credit card transactions using
Kaggle's `fraudTrain` / `fraudTest` datasets and a combination of Isolation
Forest (unsupervised) and RandomForest (supervised) models.

## Project structure

- `runone.ipynb` – feature engineering and model development on `fraudTrain`.
- `transaction-risk-intelligence-run.ipynb` – cleaned Kaggle version of the Phase 1 notebook.
- Kaggle notebook: https://www.kaggle.com/code/hritwikbali/transaction-risk-intelligence-run
- `features_phase1.py` – reusable feature engineering functions for batch and API scoring.
- `models_phase1.py` – model loading and prediction utilities.
- `batch_feature_and_score.py` – offline batch scoring script.
- `services/` – business logic (e.g. scoring service).
- `app/` – FastAPI app exposing scoring endpoints.
- `db_config.py`, `load_csv_to_mysql.py` – optional MySQL integration helpers.
- `iforest_phase1_final.joblib`, `iforest_scaler_phase1.joblib`, `rf_phase1_final.joblib` – trained models and scaler artifacts.

## Data

The original datasets are too large for GitHub (>100 MB), so they are **not**
stored in this repository.

You can download them directly from Kaggle:

- Dataset: `Credit Card Fraud Detection` by `Kartik2112`  
- Files: `fraudTrain.csv`, `fraudTest.csv`

After downloading, place them like this:

```text
project-root/
  archive (1)/
    fraudTrain.csv
    fraudTest.csv
```

(or adjust the paths in the notebooks / scripts to match your local layout).

## Environment & setup

```bash
# create & activate env (example with conda)
conda create -n tri-platform python=3.13
conda activate tri-platform

pip install -r requirements.txt  # (add this file later if needed)
```

Key Python packages:

- pandas, numpy  
- scikit-learn  
- matplotlib  
- joblib  
- fastapi, uvicorn (for the API)  
- mysql-connector-python or similar (if using MySQL)

## Running the notebooks

1. Open `runone.ipynb` or `transaction-risk-intelligence-run.ipynb` in Jupyter or VS Code.
2. Make sure the Kaggle CSVs are available locally (see **Data** section).
3. Run cells in order to reproduce feature engineering and model training
   for Phase 1.

## API scoring service (FastAPI)

From the project root:

```bash
uvicorn app.main:app --reload
```

This starts a local scoring API that uses the saved Phase 1 models to score new
transactions.

## Batch scoring

To run offline batch scoring on a CSV:

```bash
python batch_feature_and_score.py \
  --input-path path/to/input.csv \
  --output-path path/to/scored_output.csv
```

The script will:

1. Apply the same Phase 1 feature pipeline.
2. Load trained models from the `.joblib` files.
3. Write predictions and fraud scores to the output CSV.
```

Then in your terminal:

```bash
git add README.md
git commit -m "Polish README formatting"
git push
```
