import pickle, os
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import streamlit as st

CAL_MODEL_PATH = "models/calibrated_ensemble.pkl"

def calibrate_and_save(base_model, X_calib, y_calib):
    cal = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    cal.fit(X_calib, y_calib)
    with open(CAL_MODEL_PATH, "wb") as f:
        pickle.dump(cal, f)
    return cal

def load_calibrated_model():
    if not os.path.exists(CAL_MODEL_PATH):
        return None
    try:
        if os.path.getsize(CAL_MODEL_PATH) == 0:
            return None
        with open(CAL_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def plot_reliability_diagram(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids, frac_pos = [], []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_mids.append(y_prob[mask].mean())
            frac_pos.append(y_true[mask].mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_mids, frac_pos, "o-", color="#1D9E75", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    st.pyplot(fig)
    plt.close()