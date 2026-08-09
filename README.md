# Telecom Churn — end-to-end MLOps

Predicts which telecom customers will leave, then serves that model as an API,
gives it a web app, tests it in CI, and retrains it on a schedule.

| | |
|---|---|
| **Dataset** | IBM Telco Customer Churn — 7,043 customers, 21 columns, 26.5% churn |
| **Best model** | Logistic Regression (`class_weight="balanced"`) |
| **ROC-AUC** | **0.8416** (PR-AUC 0.6327, F1 0.6136) |
| **Stack** | scikit-learn · MLflow · FastAPI · Streamlit · Docker · GitHub Actions |
| **Hosting** | Render (API) + Streamlit Community Cloud (UI) |

**Live app:** _paste your Streamlit URL here_
**API docs:** _paste your Render URL + `/docs` here_

---

## Architecture

```
data/telco_churn.csv
        │
        ▼
src/preprocess.py ──┐   (the SAME pipeline for training and serving)
        │           │
        ▼           │
   src/train.py ────┘──► MLflow (compare runs) ──► models/model.pkl
                                                        │
                                    ┌───────────────────┴────────────┐
                                    ▼                                ▼
                         api/main.py (FastAPI)  ◄── HTTP ──  dashboard.py (Streamlit)
                                    │
                              Render (Docker)          Streamlit Community Cloud

GitHub Actions:  ci.yml → test on every push   ·   retrain.yml → retrain weekly
```

---

## Run it locally

```bash
git clone <your-repo-url>
cd churn-mlops

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**1. Train (also creates `models/model.pkl`)**

```bash
python src/train.py
# logreg         ROC-AUC=0.8416  PR-AUC=0.6327  F1=0.6136
# random_forest  ROC-AUC=0.8227  PR-AUC=0.6131  F1=0.5372
# BEST: logreg  (ROC-AUC=0.8416)  ->  saved models/model.pkl
```

**2. Compare the runs in MLflow**

```bash
mlflow ui        # http://127.0.0.1:5000
```

**3. Start the API**

```bash
uvicorn main:app --app-dir api --reload
# http://127.0.0.1:8000/docs
```

**4. Start the web app** (in a second terminal)

```bash
streamlit run dashboard.py
# http://127.0.0.1:8501
```

**5. Run the quality gate**

```bash
pytest -v
```

---

## API

`GET /health` → `{"status":"ok","model":"model.pkl"}`

`POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
       "tenure":5,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
       "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":89.9,"TotalCharges":450.5}'
```

```json
{"churn_probability": 0.8989, "will_churn": true, "risk": "high"}
```

A long-tenure two-year customer returns `0.0143` → `"risk":"low"`.

---

## Deploy

**API → Render** (free, no card)

1. New → Web Service → connect this repo.
2. Runtime **Docker** (the `Dockerfile` is at the repo root). Plan: **Free**.
3. Health check path: `/health`. Deploy.
4. Copy the URL, e.g. `https://churn-api-xxxx.onrender.com`.

**UI → Streamlit Community Cloud** (free)

1. New app → this repo → main file `dashboard.py`.
2. Advanced settings → Secrets:
   ```toml
   API_URL = "https://churn-api-xxxx.onrender.com"
   ```
3. Deploy.

Both hosts redeploy automatically on every push to `main`.

> The dashboard calls the API first. If the API is unreachable (Render's free
> tier sleeps after ~15 idle minutes and takes ~50s to wake), it falls back to
> scoring with the same `models/model.pkl` locally, so the demo never breaks.

---

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `DATA_PATH` | `data/telco_churn.csv` | `src/train.py`, tests |
| `MODEL_PATH` | `models/model.pkl` | `src/train.py`, `api/main.py`, `dashboard.py` |
| `API_URL` | `http://127.0.0.1:8000` | `dashboard.py` |
| `PORT` | `8000` | Docker / Render |

---

## Why this is MLOps, not a notebook

- **Class imbalance** — only 26.5% churn, so plain accuracy lies. The models use
  `class_weight="balanced"` and are judged on ROC-AUC / PR-AUC.
- **No train/serve skew** — `src/preprocess.py` builds one `ColumnTransformer`
  that is fitted during training and stored *inside* `model.pkl`, so the API
  transforms data exactly the way training did.
- **Experiment tracking** — MLflow logs params, metrics and the model artifact
  for every run, so "which model is best" is evidence, not opinion.
- **Quality gate in CI** — `tests/test_pipeline.py` fails the build if ROC-AUC
  drops below 0.78 or if the high-risk/low-risk ordering breaks.
- **Automated retraining** — `retrain.yml` retrains weekly, re-runs the gate,
  and commits the new `model.pkl`, which triggers a redeploy. That closed loop
  is the thing that makes it MLOps.

---

## Repo layout

```
churn-mlops/
├─ data/telco_churn.csv          the real dataset (7,043 customers)
├─ src/
│  ├─ preprocess.py              clean + feature pipeline (train AND serve)
│  └─ train.py                   train, track in MLflow, save the best model
├─ api/
│  ├─ schema.py                  typed input (one customer)
│  └─ main.py                    FastAPI: /predict, /health
├─ dashboard.py                  Streamlit web app
├─ tests/test_pipeline.py        CI quality gate
├─ models/model.pkl              the saved model (created by train.py)
├─ Dockerfile
├─ render.yaml
├─ requirements.txt
└─ .github/workflows/
   ├─ ci.yml                     test on every push
   └─ retrain.yml                retrain weekly
```
