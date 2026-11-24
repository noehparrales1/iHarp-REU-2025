"""
LSTM-IDRIFTNET inference.

`run_prediction(file_2024, file_2025)` is called directly by the dashboard.
Running this file starts the original standalone Flask API (POST /predict).
"""
import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic

BASE_DIR = pathlib.Path(__file__).resolve().parent
CSV_FOLDER = BASE_DIR / "CSV_LSTM_IDRIFTNET"
MODEL_WEIGHTS = BASE_DIR / "Model_LSTM_IDRIFTNET" / "lstm_idriftnet.weights.h5"

FEATURE_COLS = ['Latitude', 'Longitude', 'Area', 'u10', 'v10', 'uo', 'vo']
SEQ_LEN = 5


# ===========================
# Physics-informed residual
# ===========================
rho_air, rho_water, rho_ice = 1.2, 1027, 850
Ca, Cw, f = 0.9, 1.3, 1e-4
_g = tf.sqrt((rho_air*(rho_water-rho_ice))/(rho_water*rho_ice)*(Ca/Cw))


@tf.function
def physics_model(tw):
    u10, v10, uo, vo, area, lat_p, lon_p = tf.split(tw, 7, axis=-1)
    va = tf.concat([u10, v10], -1)
    vw = tf.concat([uo, vo], -1)
    va_mag = tf.norm(va, axis=-1, keepdims=True)
    S = tf.where(area > 0, area*1e6/(2*tf.sqrt(area*1e6)), tf.zeros_like(area))
    L = tf.where(S > 0, _g*Cw/(np.pi*f)*va_mag/S, tf.zeros_like(S))
    L4 = L**4; sq = tf.sqrt(1.+4.*L4)
    a = tf.where(L > 0, 1./(2.*L**3)*(sq-1.), tf.zeros_like(L))
    b = tf.where(L > 0, 1./(tf.sqrt(2.)*L**3)*tf.sqrt(tf.maximum((1.+L4)*sq-3.*L4-1., 0)), tf.zeros_like(L))
    k_cross = tf.concat([-va[..., 1:2], va[..., 0:1]], -1)
    wind = _g*(-a*k_cross + b*va)
    vi = (vw + wind)*0.8
    lat = lat_p + vi[..., 1:2]*0.01
    lon = lon_p + vi[..., 0:1]*0.01
    return tf.concat([lat, lon], -1)


# ===========================
# Model
# ===========================
class LSTM_IDRIFTNET(tf.keras.Model):
    def __init__(self, units=64):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(units)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        x_seq, tw = inputs
        x = self.lstm(x_seq)
        x = self.lstm2(x)
        residual = physics_model(tw)
        x = tf.concat([x, tw], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        delta = self.out(x)
        return residual + delta


_model = None


def get_model():
    """Build the model and load the trained weights once, then reuse it."""
    global _model
    if _model is None:
        model = LSTM_IDRIFTNET()
        # Run one dummy forward pass so every sub-layer is built. In Keras 3,
        # model.build() alone leaves the LSTM/Dense layers unbuilt, so the
        # saved weights would not be loaded into them.
        model((tf.zeros((1, SEQ_LEN, len(FEATURE_COLS))), tf.zeros((1, 7))), training=False)
        model.load_weights(str(MODEL_WEIGHTS))
        _model = model
    return _model


# ===========================
# Inference
# ===========================
def _read_year(source, year):
    df = pd.read_excel(source)
    missing = [c for c in ['Date'] + FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{year} file is missing column(s): {', '.join(missing)}")
    df = df[['Date'] + FEATURE_COLS].dropna()
    df['Year'] = year
    df['Date'] = pd.to_datetime(df['Date'])
    return df.reset_index(drop=True)


def _build_seq(df):
    X, y, tw, dates = [], [], [], []
    for i in range(len(df) - SEQ_LEN):
        X.append(df[FEATURE_COLS].iloc[i:i+SEQ_LEN].values)
        y.append(df[['Latitude', 'Longitude']].iloc[i+SEQ_LEN].values)
        tw.append(df.iloc[i+SEQ_LEN][['u10', 'v10', 'uo', 'vo', 'Area']].tolist() +
                  df[['Latitude', 'Longitude']].iloc[i+SEQ_LEN-1].tolist())
        dates.append(df.Date.iloc[i+SEQ_LEN])
    return np.array(X, np.float32), np.array(y, np.float32), np.array(tw, np.float32), dates


def run_prediction(file_2024, file_2025):
    """
    file_2024 / file_2025: path or file-like object for an .xlsx with columns
    Date, Latitude, Longitude, Area, u10, v10, uo, vo.
    Returns metrics plus the predicted vs. ground-truth trajectory.
    """
    df_val = _read_year(file_2024, 2024)
    df_test = _read_year(file_2025, 2025)

    inference_df = pd.concat([df_val.iloc[45:], df_test.iloc[:15]], ignore_index=True)
    if len(inference_df) <= SEQ_LEN:
        raise ValueError(
            f"Not enough rows to predict: need more than {SEQ_LEN} rows combined from "
            "row 46 onward of the 2024 file and the first 15 rows of the 2025 file."
        )

    scaler = MinMaxScaler()
    scaler.fit(inference_df[FEATURE_COLS])
    inference_df_scaled = inference_df.copy()
    inference_df_scaled[FEATURE_COLS] = scaler.transform(inference_df[FEATURE_COLS])

    X_inf, y_inf, tw_inf, dates_inf = _build_seq(inference_df_scaled)
    model = get_model()

    # autoregressive rollout: feed each prediction back into the input window
    preds, x_seq = [], X_inf[:1]
    for i in range(len(tw_inf)):
        pred = model((x_seq, tf.convert_to_tensor(tw_inf[i:i+1], tf.float32)), training=False).numpy()[0]
        preds.append(pred)
        x_new = np.vstack([x_seq[0, 1:], np.hstack([pred, tw_inf[i][2:]])])
        x_seq = tf.convert_to_tensor([x_new], dtype=tf.float32)
    final_pred = np.array(preds)

    zp = np.zeros((len(final_pred), len(FEATURE_COLS)-2))
    pred_inv = scaler.inverse_transform(np.hstack([final_pred, zp]))[:, :2]
    gt_inv = scaler.inverse_transform(np.hstack([y_inf, zp]))[:, :2]
    lat_p, lon_p = pred_inv[:, 0], pred_inv[:, 1]
    lat_t, lon_t = gt_inv[:, 0], gt_inv[:, 1]

    ade_deg = np.mean(np.sqrt((lat_p-lat_t)**2 + (lon_p-lon_t)**2))
    fde_deg = np.sqrt((lat_p[-1] - lat_t[-1])**2 + (lon_p[-1] - lon_t[-1])**2)
    geo = np.array([geodesic((lat_t[i], lon_t[i]), (lat_p[i], lon_p[i])).meters for i in range(len(lat_p))])

    out_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates_inf],
        'GT_Lat': lat_t,
        'GT_Lon': lon_t,
        'Pred_Lat': lat_p,
        'Pred_Lon': lon_p
    })
    return {
        "ADE (deg)": round(float(ade_deg), 6),
        "FDE (deg)": round(float(fde_deg), 6),
        "ADE (m)": round(float(geo.mean()), 2),
        "FDE (m)": round(float(geo[-1]), 2),
        "trajectory": out_df.to_dict(orient="records"),
    }


# ===========================
# Optional standalone Flask API
# ===========================
if __name__ == '__main__':
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            result = run_prediction(request.files['file_2024'], request.files['file_2025'])
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 400
        out_csv = CSV_FOLDER / 'LSTM_IDRIFTNET_2024_2025_Inference.csv'
        pd.DataFrame(result["trajectory"]).to_csv(out_csv, index=False)
        print(f"Saved inference predictions → {out_csv}")
        return jsonify(result)

    app.run(debug=True)
