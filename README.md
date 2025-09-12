# iHARP REU 2025 – Iceberg UI Project

This repository contains research and development work completed during the **NSF iHARP REU (Institute for Harnessing Data and Model Revolution in Polar Science)**. The project applies **data science, machine learning, and visualization** to study iceberg drift and polar environments, focusing on building interactive tools for researchers.

## Features
- Flask backend API for running machine learning predictions on iceberg drift.
- Integration with **Google Cloud Storage** for downloading model weights securely.
- Frontend dashboard for uploading input Excel files and visualizing predictions.
- LSTM-based deep learning model (IDRIFTNET) for trajectory forecasting.
- Outputs include trajectory predictions and error metrics (ADE/FDE) in both degrees and meters.

## Requirements
Install dependencies with:
pip install -r requirements.txt


## Running the App

### 1. Create a virtual environment and install dependencies
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

### 2. Set your Google Cloud credentials
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\secure\keys\iceberg-sa.json"
# Optional: make it permanent for new terminals
# setx GOOGLE_APPLICATION_CREDENTIALS "C:\secure\keys\iceberg-sa.json"

### 3. Run Backend 
python backend/predict.py

### 4. Run Frontend
. .venv\Scripts\Activate.ps1
python frontend/dashboard.py

Project Context

This work contributes to iHARP’s broader mission of harnessing data-driven methods to better understand and model polar processes. By combining machine learning with environmental datasets, the project provides tools for analyzing iceberg drift, which is critical for both scientific study and practical navigation in polar regions.
