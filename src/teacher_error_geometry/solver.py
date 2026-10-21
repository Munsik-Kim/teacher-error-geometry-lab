"""The learner's information boundary is exactly (Z,T)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Fit:
    w: float
    rank: int
    status: str
    error: str = ""
    solver_called: bool = False


def fit_student(z, target):
    """Unregularized, no-intercept FP64 OLS; failures remain explicit."""
    try:
        z = np.asarray(z, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if z.ndim != 1 or target.shape != z.shape or z.size == 0:
            return Fit(float("nan"), -1, "invalid_shape")
        if not np.isfinite(z).all() or not np.isfinite(target).all():
            return Fit(float("nan"), -1, "nonfinite_input")
    except (TypeError, ValueError) as exc:
        return Fit(float("nan"), -1, "invalid_input", str(exc))
    try:
        w, _, rank, _ = np.linalg.lstsq(z[:, None], target, rcond=None)
        if rank != 1:
            return Fit(float("nan"), int(rank), "rank_deficient", solver_called=True)
        if not np.isfinite(w).all():
            return Fit(float("nan"), int(rank), "nonfinite_coefficient", solver_called=True)
        return Fit(float(w[0]), int(rank), "ok", solver_called=True)
    except Exception as exc:
        return Fit(float("nan"), -1, "solver_exception", repr(exc), True)
