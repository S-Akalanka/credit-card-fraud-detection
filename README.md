# Credit Card Fraud Detection
Production-grade ML pipeline - XGBoost · Optuna · MLflow · FastAPI

## Dataset
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 transactions · 0.17% fraud · 30 features (V1–V28 PCA + Amount + Time)

```
data/
├── raw/
│   └── creditcard.csv       <- drop raw dataset here
└── processed/
    └── features_latest.parquet   <- auto-generated
```

---

## Stack
| Tool | Purpose |
|---|---|
| XGBoost | Primary model |
| Optuna | Hyperparameter tuning (TPE Bayesian) |
| MLflow | Experiment tracking + model registry |
| FastAPI | REST API serving |
| Pydantic | Config validation + request schemas |

---

## Project structure
```
fraud_prod/
├── configs/config.yaml          <- all hyperparams and settings
├── src/
│   ├── utils/
│   │   ├── config.py            <- Pydantic config validation
│   │   ├── logger.py            <- structured logging
│   │   └── metrics.py           <- PR-AUC, F1, threshold finding
│   ├── features/
│   │   ├── engineer.py          <- Hour, Amount_log feature engineering
│   │   └── store.py             <- parquet feature store
│   ├── pipelines/
│   │   ├── ingest.py            <- load from feature store + stratified split
│   │   ├── preprocess.py        <- model-aware scaling + SMOTE
│   │   ├── train.py             <- all model + strategy MLflow runs
│   │   └── evaluate.py          <- best model evaluation + plots
│   ├── models/
│   │   ├── base.py              <- abstract model interface
│   │   └── classifiers.py       <- Baseline, LR, RF, XGBoost, LightGBM
│   ├── tuning/
│   │   └── tuner.py             <- Optuna TPE tuning for XGBoost
│   └── serving/
│       ├── predictor.py         <- ML inference logic
│       └── fastapi_app.py       <- HTTP routing only
├── notebooks/
│   ├── eda.ipynb                <- exploratory data analysis
│   └── evaluate.ipynb           <- interactive threshold tuning
```

---

## Run order

### 1. Setup
```bash
pip install -r requirements.txt
# drop creditcard.csv into data/raw/
```

### 2. Start MLflow
```bash
make mlflow
# UI at http://localhost:5000
```

### 3. Build feature store
```bash
make features
# engineers Hour, Amount_log from raw data
# saves to data/processed/features_latest.parquet
```

### 4. EDA
```bash
jupyter notebook notebooks/eda.ipynb
```

### 5. Train all models
```bash
make train
# runs 13 MLflow experiments:
# baseline + 4 models x 3 imbalance strategies
# sort by val_pr_auc in MLflow UI to find winner
```

### 6. Hyperparameter tuning (optional)
```bash
make tune
# 75 Optuna trials on XGBoost
# logs nested runs under xgboost__optuna_tuning
```

### 7. Evaluate winner
```bash
jupyter notebook notebooks/evaluate.ipynb
# change SELECTED_RUN
# find best threshold interactively
# log threshold back to MLflow in final cell
```

### 8. Serve
```bash
make serve
# API at http://localhost:8000
# docs at http://localhost:8000/docs
```

---

## API usage

### Health check
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model": "xgboost__none", "threshold": 0.847}
```

### Predict
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
         "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
         "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
         "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
         "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
         "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
         "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02,
         "Amount": 149.62, "Hour": 2.5, "Amount_log": 5.01
       }
     }'
```