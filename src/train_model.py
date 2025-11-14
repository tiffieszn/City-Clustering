import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

DATA_PATH = os.environ.get("DATA_PATH", "./data/city_lifestyle_dataset.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def preprocess(df, features=None):
    if features is None:
        features = [
            'population_density', 'avg_income', 'internet_penetration', 'avg_rent', 
            'air_quality_index', 'public_transport_score',
            'happiness_score', 'green_space_ratio'
        ]
    x = df[features].copy()
    x = x.fillna(x.median())
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    return xs, scaler, features

def find_k(x, k_min=2, k_max=8):
    scores = {}
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(x)
        score = silhouette_score(x, labels)
        scores[k] = score
    return scores

def train_kmeans(x, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(x)
    return km, labels

if __name__ == '__main__':
    df = load_data()
    xs, scaler, features = preprocess(df)
#best k 
    scores = find_k(xs, 2, 8)
    print('Silhouette scores by k:', scores)
#default k=4 
    k = 4
    km, labels = train_kmeans(xs, n_clusters=k)
#PCA 2D visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(xs)
#labels + coords to df
    df_out = df.copy()
    df_out['cluster'] = labels
    df_out['pca_x'] = coords[:,0]
    df_out['pca_y'] = coords[:,1]
#save model artifacts
    joblib.dump(km, os.path.join(MODEL_DIR, 'kmeans_model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'), compress=3)
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca.joblib'))
    df_out.to_csv(os.path.join(MODEL_DIR, 'clustered_cities.csv'), index=False)
#save cluster centers on og ft scale
    centers_scaled = km.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers, columns=features)
    centers_df['cluster'] = range(len(centers_df))
    centers_df.to_csv(os.path.join(MODEL_DIR, 'cluster_centers.csv'), index=False)

    print('Saved models to', MODEL_DIR)