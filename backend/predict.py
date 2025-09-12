from flask import Flask, request, jsonify
import os, numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
from google.cloud import storage
import pathlib

app = Flask(__name__)

# Set the environment variable that tells Google where to find the credentials JSON file
# This is needed to authenticate with Google Cloud services (like GCS)
cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not cred_path or not os.path.exists(cred_path):
    raise RuntimeError(
        "Set GOOGLE_APPLICATION_CREDENTIALS to your service-account JSON path."
    )


def download_weights_from_gcs(bucket_name: str,
                              blob_name: str,
                              local_path: str,
                              force: bool = False) -> None:
    """Parameters:
    - bucket_name: name of your GCS bucket (e.g., 'iceberg_ui_model_storage')
    - blob_name: the name of the file in the bucket (e.g., 'lstm_idriftnet.weights.h5')
    - local_path: where to save it on your computer/server (e.g., './models/lstm_idriftnet.weights.h5')
    - force: if True, re-download the file even if it already exists locally
    """
    
    """
    Pulls <blob_name> from GCS bucket to <local_path>.
    • Skips download if the file already exists unless force=True.
    """
    local_path = pathlib.Path(local_path)

    client  = storage.Client()
    bucket  = client.bucket(bucket_name)
    blob    = bucket.blob(blob_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder
    blob.download_to_filename(str(local_path))
    print(f"✅  Downloaded '{blob_name}' → '{local_path}'")

#frontend send excel files here when user clicks Run ML Prediction
@app.route('/predict', methods=['POST'])
def predict():
    # file upload paths
    csv_folder = "CSV_LSTM_IDRIFTNET"
    model_folder = "Model_LSTM_IDRIFTNET"
    model_weights = os.path.join(model_folder, "lstm_idriftnet.weights.h5")


    download_weights_from_gcs(
        bucket_name="iceberg_ui_model_storage",               
        blob_name="lstm_idriftnet.weights.h5",        
        local_path=model_weights,
        force=False                                   
    )
    
    #creates files if they dont exist already
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    # save uploaded files from request
    file_2024 = request.files['file_2024']
    file_2025 = request.files['file_2025']
    
    #save locally
    file_2024.save(os.path.join(csv_folder, "A23A_2024.xlsx"))
    file_2025.save(os.path.join(csv_folder, "A23A_2025.xlsx"))

    # load data
    feature_cols = ['Latitude','Longitude','Area','u10','v10','uo','vo']
    df_2024 = pd.read_excel(os.path.join(csv_folder, "A23A_2024.xlsx"))[['Date'] + feature_cols].dropna()
    df_2025 = pd.read_excel(os.path.join(csv_folder, "A23A_2025.xlsx"))[['Date'] + feature_cols].dropna()

    df_2024['Year'] = 2024
    df_2025['Year'] = 2025
    df_2024['Date'] = pd.to_datetime(df_2024['Date'])
    df_2025['Date'] = pd.to_datetime(df_2025['Date'])

    df_val = df_2024.reset_index(drop=True)
    df_test = df_2025.reset_index(drop=True)
    inference_df = pd.concat([df_val.iloc[45:], df_test.iloc[:15]], ignore_index=True)

    scaler = MinMaxScaler()
    scaler.fit(inference_df[feature_cols])
    inference_df_scaled = inference_df.copy()
    inference_df_scaled[feature_cols] = scaler.transform(inference_df[feature_cols])

    seq_len = 5
    def build_seq(df):
        X, y, tw, dates = [], [], [], []
        for i in range(len(df) - seq_len):
            X.append(df[feature_cols].iloc[i:i+seq_len].values)
            y.append(df[['Latitude','Longitude']].iloc[i+seq_len].values)
            tw.append(df.iloc[i+seq_len][['u10','v10','uo','vo','Area']].tolist() +
                      df[['Latitude','Longitude']].iloc[i+seq_len-1].tolist())
            dates.append(df.Date.iloc[i+seq_len])
        return np.array(X, np.float32), np.array(y, np.float32), np.array(tw, np.float32), dates

    X_inf, y_inf, tw_inf, dates_inf = build_seq(inference_df_scaled)

    rho_air, rho_water, rho_ice = 1.2, 1027, 850
    Ca, Cw, f = 0.9, 1.3, 1e-4
    _g = tf.sqrt((rho_air*(rho_water-rho_ice))/(rho_water*rho_ice)*(Ca/Cw))

    @tf.function
    def physics_model(tw):
        u10,v10,uo,vo,area,lat_p,lon_p = tf.split(tw, 7, axis=-1)
        va = tf.concat([u10,v10], -1)
        vw = tf.concat([uo,vo], -1)
        va_mag = tf.norm(va, axis=-1, keepdims=True)
        S = tf.where(area>0, area*1e6/(2*tf.sqrt(area*1e6)), tf.zeros_like(area))
        L = tf.where(S>0, _g*Cw/(np.pi*f)*va_mag/S, tf.zeros_like(S))
        L4 = L**4; sq = tf.sqrt(1.+4.*L4)
        a = tf.where(L>0, 1./(2.*L**3)*(sq-1.), tf.zeros_like(L))
        b = tf.where(L>0, 1./(tf.sqrt(2.)*L**3)*tf.sqrt(tf.maximum((1.+L4)*sq-3.*L4-1.,0)), tf.zeros_like(L))
        k_cross = tf.concat([-va[...,1:2], va[...,0:1]], -1)
        wind = _g*(-a*k_cross + b*va)
        vi = (vw + wind)*0.8
        lat = lat_p + vi[...,1:2]*0.01
        lon = lon_p + vi[...,0:1]*0.01
        return tf.concat([lat, lon], -1)

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

    model = LSTM_IDRIFTNET()
    model.build([(None, seq_len, len(feature_cols)), (None, 7)])
    model.load_weights(model_weights)
   

    def predict_seq(x_seq_init, tw_seq):
        preds, x_seq = [], x_seq_init[:1]
        for i in range(len(tw_seq)):
            pred = model((x_seq, tf.convert_to_tensor(tw_seq[i:i+1], tf.float32)), training=False).numpy()[0]
            preds.append(pred)
            x_new = np.vstack([x_seq[0,1:], np.hstack([pred, tw_seq[i][2:]])])
            x_seq = tf.convert_to_tensor([x_new], dtype=tf.float32)
        return np.array(preds)

    final_pred = predict_seq(X_inf, tw_inf)
    zp = np.zeros((len(final_pred), len(feature_cols)-2))
    pred_inv = scaler.inverse_transform(np.hstack([final_pred, zp]))[:, :2]
    gt_inv = scaler.inverse_transform(np.hstack([y_inf, zp]))[:, :2]
    lat_p, lon_p = pred_inv[:,0], pred_inv[:,1]
    lat_t, lon_t = gt_inv[:,0], gt_inv[:,1]

    ade_deg = np.mean(np.sqrt((lat_p-lat_t)**2 + (lon_p-lon_t)**2))
    fde_deg = np.sqrt((lat_p[-1] - lat_t[-1])**2 + (lon_p[-1] - lon_t[-1])**2)
    geo = np.array([geodesic((lat_t[i], lon_t[i]), (lat_p[i], lon_p[i])).meters for i in range(len(lat_p))])
    ade_m = np.mean(geo)
    fde_m = geodesic((lat_t[-1], lon_t[-1]), (lat_p[-1], lon_p[-1])).meters

    # Save CSV
    out_df = pd.DataFrame({
    'Date': dates_inf,
    'GT_Lat': lat_t,
    'GT_Lon': lon_t,
    'Pred_Lat': lat_p,
    'Pred_Lon': lon_p
})
    output_csv = os.path.join(csv_folder, 'LSTM_IDRIFTNET_2024_2025_Inference.csv')
    out_df.to_csv(output_csv, index=False)
    print(f"Saved inference predictions → {output_csv}")

    os.remove(model_weights)
    return jsonify({
        "ADE (deg)": round(ade_deg, 6),
        "FDE (deg)": round(fde_deg, 6),
        "ADE (m)": round(ade_m, 2),
        "FDE (m)": round(fde_m, 2),
        "trajectory": out_df.to_dict(orient="records")
    })

if __name__ == '__main__':
    app.run(debug=True)
