# Credit Risk Assessment using SHAP

A machine learning web app that predicts the probability of loan default and returns a **High Risk / Low Risk** verdict. The model is an XGBoost classifier (benchmarked against logistic regression), with an F1-optimized decision threshold and SHAP-based explainability. It is served through a FastAPI backend with a lightweight web frontend.

## Features

- Loan default prediction from applicant and loan details
- Model comparison: logistic regression vs. XGBoost
- Decision threshold tuned on the precision-recall curve (best F1)
- SHAP analysis to explain which features drive risk
- FastAPI service with input validation and auto-generated docs
- Static frontend with a probability gauge and risk verdict
- One-file deployment config for Render

## Project Structure

```
├── main.py                  # FastAPI app (API + static file serving)
├── Credit_Risk.ipynb        # EDA, training, threshold tuning, SHAP
├── credit_risk_dataset.csv  # Training data
├── credit_risk_model.pkl    # Trained model pipeline
├── best_threshold.pkl       # Tuned decision threshold
├── static/                  # Frontend (index.html, style.css, script.js)
├── requirements.txt
├── runtime.txt              # Python version for deployment
└── render.yaml              # Render deployment config
```

## Getting Started

**Requirements:** Python 3.11

```bash
# Clone
git clone https://github.com/RishabPugalia/Credit-Risk-Assessment.git
cd Credit-Risk-Assessment

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** for the web app, or **http://127.0.0.1:8000/docs** for the interactive API docs.

> The pickled model was trained with `scikit-learn==1.6.1`. Keep the pinned versions in `requirements.txt` to avoid unpickling errors.

## API

### `POST /predict`

**Request**

```json
{
  "person_age": 30,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 4,
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

**Response**

| Field | Description |
|---|---|
| `default_probability` | Predicted probability of default |
| `default_prediction` | 1 if probability ≥ threshold, else 0 |
| `threshold` | Tuned decision threshold |
| `Result` | `"High Risk"` or `"Low Risk"` |

## Model Training

Training and analysis live in `Credit_Risk.ipynb`:

1. Exploratory data analysis and cleaning
2. Preprocessing: imputation, scaling for numeric features, one-hot encoding for categoricals
3. Logistic regression baseline and XGBoost model
4. Threshold selection by maximizing F1 on the precision-recall curve
5. SHAP explainability for global and per-prediction feature impact

To run the notebook:

```bash
pip install jupyter shap matplotlib seaborn
jupyter notebook Credit_Risk.ipynb
```

## Deployment

The repo includes a `render.yaml` for [Render](https://render.com). Connect the repository, and the service starts with:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Tech Stack

Python, FastAPI, scikit-learn, XGBoost, SHAP, pandas, HTML/CSS/JavaScript
