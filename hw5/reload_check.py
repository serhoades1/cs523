
import joblib
import numpy as np
import pandas as pd

loaded_pipe = joblib.load("dropout_tuned_pipeline.joblib")
X_500 = pd.read_csv("sample_500.csv")
reloaded_preds = loaded_pipe.predict_proba(X_500)[:, 1]
notebook_preds = np.load("notebook_preds_500.npy")

max_abs_diff = float(np.max(np.abs(reloaded_preds - notebook_preds)))
print(f"Max absolute difference in fresh process: {max_abs_diff:.15f}")
assert max_abs_diff < 1e-12
