from pydantic import BaseModel
from typing import List, Dict

class VoiceFeatures(BaseModel):
    mdvp_fo: float
    mdvp_fhi: float
    mdvp_flo: float
    mdvp_jitter_pct: float
    mdvp_jitter_abs: float
    mdvp_rap: float
    mdvp_ppq: float
    jitter_ddp: float
    mdvp_shimmer: float
    mdvp_shimmer_db: float
    shimmer_apq3: float
    shimmer_apq5: float
    mdvp_apq: float
    shimmer_dda: float
    nhr: float
    hnr: float
    rpde: float
    dfa: float
    spread1: float
    spread2: float
    d2: float
    ppe: float

class PredictionResponse(BaseModel):
    prediction: str
    risk_label: str
    probability_pd: float
    confidence: str
    disclaimer: str