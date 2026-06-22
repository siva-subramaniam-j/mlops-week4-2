# Iris Classifier MLOps Assignment

This folder contains the final implementation of the MLOps weekly assignment for building and deploying an Iris classification pipeline using scikit-learn and Google Cloud Storage.

## Assignment Overview

This assignment implements a complete machine learning pipeline that:
- **Task 3**: Stores training data (raw + versioned) in Google Cloud Storage (GCS)
- **Task 4**: Executes an Iris training pipeline using sklearn Pipeline (StandardScaler + DecisionTreeClassifier)
- **Task 5**: Runs training and inference twice with timestamped artifact organization
- **Task 6 (Optional)**: Compares results across multiple data versions (v1, v2)

## Files

- **`GA1_notebook.ipynb`** — Main Jupyter notebook containing the complete end-to-end ML pipeline
  - Data upload to GCS
  - sklearn Pipeline definition and training
  - Inference script generation
  - Training loops (2 runs on raw data, optional v1/v2 comparison)

- **`inference.py`** — Standalone inference script
  - Downloads trained model and evaluation data from GCS
  - Runs predictions on evaluation set
  - Outputs metrics (accuracy, classification report)
  - Saves predictions to CSV and metrics to JSON

- **`requirements.txt`** — Python package dependencies

## Setup

### Prerequisites
- Python 3.12+
- Google Cloud project with Storage API enabled
- GCS bucket (`gs://week1-assignment-bucket-iit`)
- Authentication: `gcloud auth application-default login`

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Authenticate with Google Cloud:
```bash
gcloud auth application-default login
```

## Usage

### Running the Notebook

1. Open `GA1_notebook.ipynb` in Jupyter/Colab/Workbench
2. Update `PROJECT_ID` and `BUCKET_URI` variables if needed (currently set to the assignment values)
3. Run cells in order:
   - Install packages
   - Set project configuration
   - Task 2: Upload datasets to GCS
   - Task 3: Define training pipeline functions
   - Task 4: Define Inference pipeline functions
   - Task 5: Train twice and run inference
   - Task 6 (Optional): Compare data versions

### Running Inference Standalone

```bash
python inference.py \
  --model-gcs-uri gs://week1-assignment-bucket-iit/runs/20260622-200648/raw/model/model.joblib \
  --eval-gcs-uri gs://week1-assignment-bucket-iit/runs/20260622-200648/raw/data/eval.csv \
  --results-dir ./inference_results
```

**Arguments:**
- `--model-gcs-uri` (required): GCS path to the trained sklearn Pipeline
- `--eval-gcs-uri` (required): GCS path to the evaluation dataset CSV
- `--results-dir` (optional, default: `inference_results`): Local directory for outputs

**Output files:**
- `predictions.csv` — Predictions on evaluation set
- `inference_metrics.json` — Accuracy and classification report

## Architecture

### Training Pipeline (sklearn)
```
StandardScaler → DecisionTreeClassifier(max_depth=3)
```

### Data Flow
```
Local Data → GCS Upload → Train/Test Split → Pipeline Training → Model Save → GCS Upload
                                                                                    ↓
                                                                          Model Artifacts (timestamped)
```

### Artifact Organization in GCS
```
gs://week1-assignment-bucket-iit/
├── data/
│   ├── raw/
│   ├── v1/
│   └── v2/
└── runs/
    └── <TIMESTAMP>/
        └── <VERSION>/
            ├── model/
            │   └── model.joblib
            ├── data/
            │   ├── train.csv
            │   └── eval.csv
            ├── metrics.json
            └── eval_predictions.csv
```

## Key Features

- **sklearn Pipeline**: Combines preprocessing (StandardScaler) with classifier in a reproducible pipeline
- **GCS Integration**: All artifacts, metrics, and data versioning stored in Google Cloud Storage
- **Timestamped Runs**: Each training execution creates a timestamped folder for reproducibility
- **Inference Script**: Standalone Python script loads model from GCS and runs batch predictions
- **Data Versioning**: Support for multiple data versions (raw, v1, v2) with separate metrics

## Configuration

Edit these variables in `GA1_notebook.ipynb` (cell 2):
```python
PROJECT_ID = "project-f9a302e4-48ab-4c0e-91b4"
LOCATION = "us-central1"
BUCKET_URI = "gs://week1-assignment-bucket-iit"
```

