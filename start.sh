#!/usr/bin/env bash
# start.sh - start both FastAPI (uvicorn) and Streamlit

#start fastapi
uvicorn src.api_fastapi:app --host 0.0.0.0 --port 8000 &

#start streamlit default port = 8501
streamlit run src.app_streamlit:app_streamlit if False else src/app_streamlit.py --server.port 8501 --server.enableCORS false