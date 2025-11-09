import argparse, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils import save_model, ensure_dirs
def collect_window(n=3000, seed=123):
    rng=np.random.default_rng(seed)
    t=rng.normal(60.0,3.2,n); h=rng.normal(35.0,4.5,n); s=rng.normal(75.0,5.5,n)
    return np.column_stack([t,h,s])
def main():
    p=argparse.ArgumentParser(); p.add_argument("--window_days",type=int,default=7); p.add_argument("--contamination",type=float,default=0.05)
    a=p.parse_args(); ensure_dirs(); X=collect_window(int(a.window_days*500)); scaler=StandardScaler().fit(X); Xz=scaler.transform(X)
    clf=IsolationForest(n_estimators=200, contamination=a.contamination, random_state=11).fit(Xz)
    save_model(clf, scaler); print("[retrain] refreshed model")
if __name__=="__main__": main()
