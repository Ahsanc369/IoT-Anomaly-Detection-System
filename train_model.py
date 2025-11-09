import argparse, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils import save_model, ensure_dirs
def synthesize_normal(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(60.0, 3.0, n); hum=rng.normal(35.0, 4.0, n); snd=rng.normal(75.0, 5.0, n)
    return np.column_stack([temp, hum, snd])
def main():
    p=argparse.ArgumentParser(); p.add_argument("--n_samples", type=int, default=6000); p.add_argument("--contamination", type=float, default=0.05)
    a=p.parse_args()
    ensure_dirs(); X=synthesize_normal(a.n_samples); scaler=StandardScaler().fit(X); Xz=scaler.transform(X)
    clf=IsolationForest(n_estimators=200, contamination=a.contamination, random_state=7).fit(Xz)
    save_model(clf, scaler); print("saved artifacts/model.pkl")
if __name__=="__main__": main()
