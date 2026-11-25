"""
Check that the NumPy model in predict.py gives the same predictions as the
Keras model in model_tf.py, on the A23a sample data.

Run from the repo root (needs TensorFlow 2.16+):
    python backend/check_numpy_matches_tf.py
"""
import numpy as np
import tensorflow as tf

from model_tf import load_tf_model
from predict import CSV_FOLDER, get_model, prepare_inputs, rollout

_, (X_inf, _, tw_inf, _) = prepare_inputs(CSV_FOLDER / "A23A_2024.xlsx", CSV_FOLDER / "A23A_2025.xlsx")

tf_model = load_tf_model()
tf_call = lambda x_seq, tw: tf_model((tf.convert_to_tensor(x_seq), tf.convert_to_tensor(tw)), training=False).numpy()

np_rollout = rollout(get_model(), X_inf, tw_inf)
tf_rollout = rollout(tf_call, X_inf, tw_inf)

max_diff = np.abs(np_rollout - tf_rollout).max()
print(f"steps: {len(np_rollout)}  max difference (scaled units): {max_diff:.2e}")
assert max_diff < 1e-4, "NumPy and TensorFlow predictions differ"
print("OK: NumPy model matches TensorFlow")
