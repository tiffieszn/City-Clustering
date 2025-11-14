import joblib
import os
import numpy as np
import pandas as pd

MODEL_DIR = os.environ.get('MODEL_DIR', './models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
KMEANS_PATH = os.path.join(MODEL_DIR, 'kmeans_model.joblib')
PCA_PATH = os.path.join(MODEL_DIR, 'pca.joblib')
CENTERS_PATH = os.path.join(MODEL_DIR, 'cluster_centers.csv')

scaler = None
kmeans = None
pca = None
centers_df = None
features = [
    'population_density', 'avg_income', 'internet_penetration',
    'avg_rent', 'air_quality_index', 'public_transport_score',
    'happiness_score', 'green_space_ratio'
]

def ensure_loaded():
    global scaler, kmeans, pca, centers_df
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    if kmeans is None:
        kmeans = joblib.load(KMEANS_PATH)
    if pca is None:
        pca = joblib.load(PCA_PATH)
    if centers_df is None:
        centers_df = pd.read_csv(CENTERS_PATH)


def predict_cluster(payload: dict):
    """payload: dict of features (names as in `features`)"""
    ensure_loaded()
    x = np.array([payload.get(f, 0) for f in features], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    cluster = int(kmeans.predict(x_scaled)[0])
    coord = pca.transform(x_scaled)[0].tolist()
    center = centers_df[centers_df['cluster'] == cluster].drop(columns=['cluster']).to_dict(orient='records')[0]
    return {
        'cluster': cluster,
        'pca_coord': coord,
        'cluster_center_features': center
    }

def get_cluster_summary():
    ensure_loaded()
    return centers_df.to_dict(orient='records')