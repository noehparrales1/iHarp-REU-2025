# NSF iHARP REU 2025  
### Iceberg Drift Forecasting & Interactive ML Dashboard

## Overview

This repository contains research conducted as part of the NSF iHARP REU program, focused on applying machine learning to polar science datasets. The primary objective was to model and forecast iceberg drift patterns using time-series data and to develop an interactive dashboard for visualizing predictions and environmental variables.

The project integrates machine learning, data engineering, and visualization to support environmental research workflows.

---

## Research Focus

- Time-series forecasting of iceberg drift trajectories  
- Application of LSTM-based deep learning models  
- Evaluation of model performance across seasonal data  
- Interactive visualization of predictions and inference outputs  

---

## Project Components

### Backend
- LSTM-based drift prediction model
- Data preprocessing and feature engineering pipeline
- Inference generation and CSV export
- Model evaluation and drift comparison analysis

### Frontend
- Interactive dashboard for:
  - Viewing drift predictions
  - Exploring inference outputs
  - Comparing seasonal patterns
- Visualization of model outputs and environmental features

---

## Methodology

- Cleaned and structured multi-year remote sensing datasets
- Built sequence-based time-series models (LSTM architecture)
- Generated inference outputs for 2024–2025 datasets
- Evaluated drift prediction accuracy
- Integrated predictions into a visualization dashboard

---

## Tools & Technologies

Python • TensorFlow / Keras • NumPy • Pandas • scikit-learn  
LSTM Networks • Time-Series Modeling  
Dash / Visualization Tools  
Data Engineering & Feature Processing

---

## Key Outcomes

- Developed a working LSTM-based drift forecasting model  
- Produced multi-season inference results  
- Built an interactive interface for research visualization  
- Strengthened applied ML workflow for environmental datasets  

---

## Live Demo

**Live demo:** https://iceberg-drift-forecasting.onrender.com  _(free hosting — the first load after a period of inactivity can take about a minute)_

Upload a 2024 and a 2025 `.xlsx` track (columns: `Date, Latitude, Longitude, Area, u10, v10, uo, vo`), or click **Run With Sample Data** to run the model on iceberg A23a. The dashboard shows predicted vs. observed positions on a map, with ADE/FDE error metrics and a per-week error table.

---

## Run Locally

```bash
pip install -r requirements.txt
python frontend/dashboard.py
# open http://127.0.0.1:8050
```

The dashboard calls the model in `backend/predict.py` directly and loads the trained weights from `backend/Model_LSTM_IDRIFTNET/`. No separate server or cloud credentials are needed.

**How inference works:** the model was built and trained in TensorFlow/Keras (`backend/model_tf.py`). For the hosted demo, `backend/predict.py` runs the same forward pass (two LSTM layers, dense layers, and the physics-based drift residual) in NumPy using the trained weights, so the app doesn't need TensorFlow and fits on a small free server. `backend/check_numpy_matches_tf.py` confirms the two implementations agree (max difference ~1e-7).

**Hosting:** deployed on Render's free tier using `render.yaml`. The free instance sleeps when idle, so the first visit after a while can take about a minute to load.

---

## Author

Noeh Parrales  
Mathematical Computer Science @ UC San Diego  
NSF iHARP REU Researcher – Data Science & Machine Learning
