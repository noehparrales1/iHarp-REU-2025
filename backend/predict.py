"""
LSTM-IDRIFTNET inference.

`run_prediction(file_2024, file_2025)` is called directly by the dashboard.

The forward pass is written in NumPy and uses the trained Keras weights from
Model_LSTM_IDRIFTNET/, so the hosted demo does not need TensorFlow. The Keras
model it mirrors is in backend/model_tf.py; backend/check_numpy_matches_tf.py
checks that both produce the same predictions.

Running this file starts the original standalone Flask API (POST /predict).
"""
import pathlib

import h5py
import numpy as np
import pandas as pd
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
_g = np.float32(np.sqrt((rho_air*(rho_water-rho_ice))/(rho_water*rho_ice)*(Ca/Cw)))


def physics_model(tw):
    """tw: (batch, 7) = u10, v10, uo, vo, area, prev_lat, prev_lon  →  (batch, 2) lat, lon"""
    u10, v10, uo, vo, area, lat_p, lon_p = np.split(tw, 7, axis=-1)
    va = np.concatenate([u10, v10], -1)
    vw = np.concatenate([uo, vo], -1)
    va_mag = np.linalg.norm(va, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(area > 0, area*1e6/(2*np.sqrt(area*1e6)), 0.0)
        L = np.where(S > 0, _g*Cw/(np.pi*f)*va_mag/S, 0.0)
        L4 = L**4
        sq = np.sqrt(1. + 4.*L4)
        a = np.where(L > 0, 1./(2.*L**3)*(sq-1.), 0.0)
        b = np.where(L > 0, 1./(np.sqrt(2.)*L**3)*np.sqrt(np.maximum((1.+L4)*sq-3.*L4-1., 0)), 0.0)
    k_cross = np.concatenate([-va[..., 1:2], va[..., 0:1]], -1)
    wind = _g*(-a*k_cross + b*va)
    vi = (vw + wind)*0.8
    lat = lat_p + vi[..., 1:2]*0.01
    lon = lon_p + vi[..., 0:1]*0.01
    return np.concatenate([lat, lon], -1)


# ===========================
# NumPy forward pass of the trained network
# ===========================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm(x_seq, kernel, recurrent_kernel, bias, return_sequences):
    """Keras LSTM (tanh / sigmoid, gate order i, f, c, o). x_seq: (batch, time, features)."""
    units = recurrent_kernel.shape[0]
    h = np.zeros((x_seq.shape[0], units), np.float32)
    c = np.zeros_like(h)
    outputs = []
    for t in range(x_seq.shape[1]):
        z = x_seq[:, t] @ kernel + h @ recurrent_kernel + bias
        i, fg, g, o = np.split(z, 4, axis=-1)
        c = _sigmoid(fg) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        outputs.append(h)
    return np.stack(outputs, axis=1) if return_sequences else h


class NumpyLSTMIDriftNet:
    """Same architecture as LSTM_IDRIFTNET in model_tf.py, loaded from the Keras weights file."""

    def __init__(self, weights_path=MODEL_WEIGHTS):
        with h5py.File(weights_path, 'r') as w:
            read = lambda path: [w[f"{path}/vars/{k}"][()] for k in sorted(w[f"{path}/vars"].keys(), key=int)]
            self.lstm1 = read("layers/lstm/cell")      # kernel, recurrent_kernel, bias
            self.lstm2 = read("layers/lstm_1/cell")
            self.fc1 = read("fc1")                     # kernel, bias
            self.fc2 = read("fc2")
            self.out = read("layers/dense_2")

    def __call__(self, x_seq, tw):
        x = _lstm(x_seq, *self.lstm1, return_sequences=True)
        x = _lstm(x, *self.lstm2, return_sequences=False)
        residual = physics_model(tw)
        x = np.concatenate([x, tw], axis=-1)
        x = np.maximum(x @ self.fc1[0] + self.fc1[1], 0)
        x = np.maximum(x @ self.fc2[0] + self.fc2[1], 0)
        delta = x @ self.out[0] + self.out[1]
        return residual + delta


_model = None


def get_model():
    """Load the trained weights once, then reuse the model."""
    global _model
    if _model is None:
        _model = NumpyLSTMIDriftNet()
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


def prepare_inputs(file_2024, file_2025):
    """Read both files and build the scaled model inputs (shared with the TF check script)."""
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
    return scaler, _build_seq(inference_df_scaled)


def rollout(model, X_inf, tw_inf):
    """Autoregressive rollout: feed each prediction back into the input window."""
    preds, x_seq = [], X_inf[:1]
    for i in range(len(tw_inf)):
        pred = np.asarray(model(x_seq, tw_inf[i:i+1]))[0]
        preds.append(pred)
        x_new = np.vstack([x_seq[0, 1:], np.hstack([pred, tw_inf[i][2:]])])
        x_seq = np.array([x_new], dtype=np.float32)
    return np.array(preds)


def run_prediction(file_2024, file_2025):
    """
    file_2024 / file_2025: path or file-like object for an .xlsx with columns
    Date, Latitude, Longitude, Area, u10, v10, uo, vo.
    Returns metrics plus the predicted vs. ground-truth trajectory.
    """
    scaler, (X_inf, y_inf, tw_inf, dates_inf) = prepare_inputs(file_2024, file_2025)
    final_pred = rollout(get_model(), X_inf, tw_inf)

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
