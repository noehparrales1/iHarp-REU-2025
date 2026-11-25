"""
LSTM-IDRIFTNET model definition (TensorFlow / Keras), as used for training.

The hosted dashboard does not import this file: backend/predict.py runs the
same forward pass in NumPy with the trained weights, so the live demo does not
need TensorFlow. This file is the reference implementation, and
backend/check_numpy_matches_tf.py confirms both give the same predictions.

Requires: pip install "tensorflow>=2.16"  (weights were saved with Keras 3)
"""
import pathlib

import numpy as np
import tensorflow as tf

MODEL_WEIGHTS = pathlib.Path(__file__).resolve().parent / "Model_LSTM_IDRIFTNET" / "lstm_idriftnet.weights.h5"
SEQ_LEN = 5
N_FEATURES = 7  # Latitude, Longitude, Area, u10, v10, uo, vo

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


def load_tf_model():
    model = LSTM_IDRIFTNET()
    # Run one dummy forward pass so every sub-layer is built. In Keras 3,
    # model.build() alone leaves the LSTM/Dense layers unbuilt, so the
    # saved weights would not be loaded into them.
    model((tf.zeros((1, SEQ_LEN, N_FEATURES)), tf.zeros((1, 7))), training=False)
    model.load_weights(str(MODEL_WEIGHTS))
    return model
