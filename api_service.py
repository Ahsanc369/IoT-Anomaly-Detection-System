from typing import List, Union
import time, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from utils import load_model, to_feature_matrix
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
app = FastAPI(title="IoT Anomaly Model", version="1.0.0")
PRED_REQUESTS = Counter("pred_requests_total","Total prediction requests",["mode"])
PRED_LATENCY = Histogram("pred_latency_seconds","Prediction latency seconds",["mode"])
class Record(BaseModel):
    machine_id: str = Field(..., examples=["A-17"])
    temperature_c: float; humidity_pct: float; sound_db: float
_model=None; _scaler=None
def ensure_loaded():
    global _model, _scaler
    if _model is None or _scaler is None:
        try: _model,_scaler=load_model()
        except FileNotFoundError as e: raise HTTPException(status_code=500, detail="Model not found. Run: python train_model.py") from e
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/metrics")
def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
@app.post("/v1/score")
def score(payload: Union[Record, List[Record]]):
    ensure_loaded(); t0=time.time()
    if isinstance(payload, list): mode="batch"; records=[r.model_dump() for r in payload]
    else: mode="single"; records=[payload.model_dump()]
    X = to_feature_matrix(records); Xz=_scaler.transform(X)
    scores = -_model.decision_function(Xz)  # higher = more anomalous
    thresh = np.quantile(scores, 0.95)     # simple batch threshold
    preds = (scores > thresh).astype(int).tolist()
    out=[{"machine_id":r["machine_id"],"anomaly_score":float(s),"is_anomaly":int(p)} for r,s,p in zip(records,scores,preds)]
    PRED_REQUESTS.labels(mode).inc(); PRED_LATENCY.labels(mode).observe(time.time()-t0)
    return out if mode=="batch" else out[0]
