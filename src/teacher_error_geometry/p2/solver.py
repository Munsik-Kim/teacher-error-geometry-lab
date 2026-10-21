"""Learner receives only assigned design/target; per-process call accounting."""
from collections import Counter
from dataclasses import dataclass
import numpy as np

COUNTS=Counter()


def counters():
    return dict(COUNTS)


@dataclass(frozen=True)
class Fit:
    coefficient: np.ndarray
    rank: int
    status: str
    error: str=""


def fit_ols(design,target):
    COUNTS["fit_requests"]+=1
    COUNTS["fit_objects"]+=1
    p=0
    try:
        x=np.asarray(design,dtype=np.float64); y=np.asarray(target,dtype=np.float64)
        p=x.shape[1] if x.ndim==2 else 0
        if x.ndim!=2 or x.shape[0]==0 or p==0 or y.shape!=(x.shape[0],):
            return Fit(np.full(p,np.nan),-1,"invalid_shape")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return Fit(np.full(p,np.nan),-1,"nonfinite_input")
        COUNTS["lstsq_calls"]+=1
        w,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
        if rank!=p:
            return Fit(np.full(p,np.nan),int(rank),"rank_deficient")
        if not np.isfinite(w).all():
            return Fit(w,int(rank),"nonfinite_coefficient")
        return Fit(w,int(rank),"ok")
    except Exception as exc:
        return Fit(np.full(p,np.nan),-1,"solver_exception",repr(exc))
