import os, pickle, numpy as np
MODEL_DIR = os.path.join("artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
def save_model(model, scaler):
    ensure_dirs()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"]
def to_feature_matrix(records):
    import numpy as np
    return np.array([[r["temperature_c"], r["humidity_pct"], r["sound_db"]] for r in records], dtype=float)
