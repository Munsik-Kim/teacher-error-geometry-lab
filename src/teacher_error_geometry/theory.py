"""Expectations over newly drawn training samples; no fitted-risk calls."""
import numpy as np
from .model import covariance


def geometry(rho, delta):
    sigma = covariance(rho)
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != (2,) or not np.isfinite(d).all():
        raise ValueError("Invalid delta")
    a = float(d[0] + rho * d[1])
    b = float(d[1] * np.sqrt(1 - rho * rho))
    mse = float(d @ sigma @ d)
    eta = None if mse == 0.0 else a * a / mse
    return {"a": a, "b": b, "teacher_mse": mse, "eta": eta}


def expectation(n, rho, delta):
    if not np.isfinite(n) or n <= 2 or int(n) != n:
        raise ValueError("Expected risk requires integer n > 2")
    g = geometry(rho, delta)
    a, b = g["a"], g["b"]
    return {"truth_risk": a * a + b * b / (n - 2),
            "teacher_risk": b * b * (1 + 1 / (n - 2))}


def eta_curve(n, c, eta):
    if not np.isfinite(n) or n <= 2 or int(n) != n:
        raise ValueError("Expected risk requires integer n > 2")
    if not np.isfinite(c) or c < 0:
        raise ValueError("Invalid c")
    if c == 0:
        return 0.0
    if eta is None or not np.isfinite(eta) or not 0 <= eta <= 1 + 1e-14:
        raise ValueError("eta must be in [0,1]")
    return c * c * (eta + (1 - eta) / (n - 2))


def risk_variance(n, rho, delta, metric):
    """Optional theoretical MC dispersion diagnostic (nondegenerate n>4)."""
    g = geometry(rho, delta)
    a, b = g["a"], g["b"]
    if b == 0:
        return 0.0
    if n <= 4:
        raise ValueError("Nondegenerate risk variance requires n > 4")
    v = b * b / (n - 2)
    result = 3 * b ** 4 / ((n - 2) * (n - 4)) - v * v
    if metric == "truth_risk":
        return result + 4 * a * a * v
    if metric == "teacher_risk":
        return result
    raise ValueError("Unknown metric")
