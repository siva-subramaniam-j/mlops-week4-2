import argparse
import json
import joblib
import pandas as pd
from google.cloud import storage
from pathlib import Path

def parse_gcs_uri(uri: str):
    assert uri.startswith('gs://'), 'URI must start with gs://'
    parts = uri[5:].split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    return bucket_name, blob_name

def download_blob(uri: str, destination: Path):
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client(project='project-f9a302e4-48ab-4c0e-91b4')
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(destination))
    print(f'Downloaded {uri} to {destination}')

def main():
    parser = argparse.ArgumentParser(description='Run inference on Iris evaluation data')
    parser.add_argument('--model-gcs-uri', required=True)
    parser.add_argument('--eval-gcs-uri', required=True)
    parser.add_argument('--results-dir', default='inference_results')
    args = parser.parse_args()

    model_path = Path(args.results_dir) / 'model.joblib'
    eval_path = Path(args.results_dir) / 'eval.csv'
    predictions_path = Path(args.results_dir) / 'predictions.csv'
    metrics_path = Path(args.results_dir) / 'inference_metrics.json'

    download_blob(args.model_gcs_uri, model_path)
    download_blob(args.eval_gcs_uri, eval_path)

    pipeline = joblib.load(model_path)  # load sklearn Pipeline
    df = pd.read_csv(eval_path)
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    y_true = df['species']
    y_pred = pipeline.predict(df[feature_cols])
    df_out = df.copy()
    df_out['prediction'] = y_pred
    df_out.to_csv(predictions_path, index=False)

    from sklearn import metrics
    accuracy = float(metrics.accuracy_score(y_true, y_pred))
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    with open(metrics_path, 'w') as f:
        json.dump({'accuracy': accuracy, 'classification_report': report}, f, indent=2)

    print(f'Inference accuracy: {accuracy:.4f}')
    print(f'Wrote predictions to {predictions_path} and metrics to {metrics_path}')

if __name__ == '__main__':
    main()

    