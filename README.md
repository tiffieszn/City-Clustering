---
title: City Clustering App
emoji: 🏙️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.33.0"
app_file: app.py
pinned: false
---

# City Lifestyle Clustering
This project clusters cities using KMeans on a city lifestyle dataset. It includes:
- Training script: `src/train_model.py`
- Model utilities: `src/model_utils.py`
- Streamlit app: `src/app_streamlit.py` (UI)
- FastAPI endpoint: `src/api_fastapi.py` (prediction API)
- Dockerfile and `start.sh` to run both services in one container

## How to run locally
1. Place the dataset at `data/city_lifestyle_dataset.csv`.
2. Create a virtualenv and install requirements:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt