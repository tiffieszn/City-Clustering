import streamlit as st
import pandas as pd
import numpy as np
from src.model_utils import predict_cluster, get_cluster_summary, features
import joblib
import os

st.set_page_config(page_title='Smart City Clustering', layout='wide')
st.title('Smart City — Lifestyle Clustering (KMeans)')
st.markdown('Upload a city CSV (or use the sample inputs) to get cluster predictions and visualize clusters.')
uploaded = st.file_uploader('Upload CSV with columns: ' + ', '.join(features), type=['csv'])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    st.write('Predicting...')
    results = []
    for _, row in df.iterrows():
        payload = {f: float(row.get(f, 0)) for f in features}
        r = predict_cluster(payload)
        results.append(r)
    res_df = pd.DataFrame(results)
    st.write(res_df)

st.sidebar.header('Single city input')
vals = {}
for f in features:
    vals[f] = st.sidebar.number_input(f, value=0.0)

if st.sidebar.button('Predict single city'):
    r = predict_cluster(vals)
    st.success(f"Assigned to cluster {r['cluster']}")
    st.json(r)

#cluster centers
st.header('Cluster centers (summary)')
centers = pd.DataFrame(get_cluster_summary())
st.dataframe(centers)

#PCA scatter (visualization)
st.header('PCA cluster visualization')
model_dir = os.environ.get('MODEL_DIR', './models')
clustered_csv = os.path.join(model_dir, 'clustered_cities.csv')
if os.path.exists(clustered_csv):
    dfc = pd.read_csv(clustered_csv)
    import plotly.express as px
    fig = px.scatter(dfc, x='pca_x', y='pca_y', color='cluster', hover_data=['city_name','country'])
    st.plotly_chart(fig)
else:
    st.info('No clustered_cities.csv found; run training to generate visualization.')