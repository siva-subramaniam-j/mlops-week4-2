# Week 4 — MLOps CI/CD Assignment

**Repository:** `mlops-week4-2`  
**Branch:** `week_4`  
**Status:** Completed

---

## What was done (simple summary)

### 1. DVC setup and data pull
- DVC was already configured from Week 2 with a **GCS remote** (`gs://week2-dvc-bucket-21f1005023/dvc`).
- Ran `dvc pull` to download versioned files into the local workspace:
  - `data/active_data.csv`
  - `data/raw/iris.csv`, `data/v1/data.csv`, `data/v2/data.csv`
  - `model/model.joblib`
- Confirmed all data and model files were present after the pull.

### 2. Tests added
- **Data validation tests** (`tests/test_data_validation.py`) — check schema, missing values, data types, and value ranges for the Iris dataset.
- **Model evaluation tests** (`tests/test_model_evaluation.py`) — load the trained model, run predictions, and check accuracy/precision/recall/F1 meet minimum thresholds.
- All tests passed locally with `pytest tests/ -v`.

### 3. GitHub Actions workflow created
- Added `.github/workflows/ci.yml` that runs automatically on every **push** and **pull request**.
- The workflow:
  1. Checks out the code
  2. Installs Python dependencies
  3. Authenticates to Google Cloud
  4. Runs `dvc pull` to fetch data and model from GCS
  5. Runs the full pytest test suite
  6. Posts a **CML report** as a comment on pull requests (pass/fail + metrics)

### 4. GCP service account and GitHub secret
- Created a dedicated service account (`dvc-github-ci`) for CI access to the GCS bucket.
- Generated a JSON key file (`gcs-dvc-key.json`).
- Added the JSON key as a GitHub Actions secret named **`GCP_SA_KEY`**.
- `GITHUB_TOKEN` was not added manually — GitHub provides it automatically for CML comments.

### 5. CI verified and working
- Pushed the `week_4` branch to GitHub.
- Opened a pull request from `week_4` → `main`.
- Confirmed the **MLOps CI** workflow ran successfully in the Actions tab.
- Reviewed the **CML bot comment** on the PR with test results and model metrics.
- Merged the pull request after all checks passed.

---

## Project structure

```
.
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── data/
│   ├── active_data.csv          # Main dataset (DVC-tracked)
│   ├── raw/iris.csv             # Source data
│   ├── v1/data.csv              # Source data
│   └── v2/data.csv              # Source data
├── model/model.joblib           # Trained model (DVC-tracked)
├── tests/                       # pytest test suite
├── scripts/                     # Data build and training scripts
├── requirements.txt
└── week_4_ci_cd_assignment.ipynb
```

---

## How to run locally

```bash
source .venv/bin/activate
pip install -r requirements.txt
dvc pull
pytest tests/ -v
```

---

## GitHub secret used

| Secret name   | What it is                                      |
|---------------|-------------------------------------------------|
| `GCP_SA_KEY`  | GCP service account JSON key for `dvc pull` in CI |

---

## Assignment tasks completed

- [x] Task 1 — Data validation tests with pytest
- [x] Task 2 — Model evaluation tests with quality gates
- [x] Task 3 — GitHub Actions workflow with DVC
- [x] Task 4 — CI runs on every push and pull request
- [x] Task 5 — CML report published on pull requests
- [x] Task 6 — Merged to main via pull request
