"""Population risk of a fixed fitted coefficient, using the quadratic form."""
import numpy as np
from .model import covariance


def fitted_risks(w, rho, delta):
    if not np.isfinite(w):
        raise ValueError("Nonfinite fitted coefficient")
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("Invalid delta")
    h = np.array([w - 1.0 - delta[0], -delta[1]], dtype=np.float64)
    return {"truth_risk": float((w - 1.0) ** 2),
            "teacher_risk": float(h @ covariance(rho) @ h)}


def conditional_teacher_risk(w, rho, delta):
    """Separate cross-check, never used to fill raw fitted risks."""
    d1, d2 = delta
    return float((w - 1.0 - d1 - rho * d2) ** 2 + d2 * d2 * (1 - rho * rho))
