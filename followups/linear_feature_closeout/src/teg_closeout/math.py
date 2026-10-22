"""Small FP64 linear audit primitives; no model or data-generator dependency."""
from __future__ import annotations

import numpy as np

ATOL = 1e-10
RTOL = 1e-8


def finite_array(value, *, ndim=None, name="array"):
    value = np.asarray(value, dtype=np.float64)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim} dimensions")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name}: nonfinite input")
    return value


def design(features):
    z = finite_array(features, ndim=2, name="features")
    if z.shape[0] == 0:
        raise ValueError("features: no rows")
    return np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))


def fit_student(features, target):
    """Learner sees only features and its assigned one-dimensional target.

    Intercept is the first coefficient. NumPy's rcond=None convention is
    retained; rank deficiency is explicitly reported, never repaired.
    """
    a = design(features)
    t = finite_array(target, ndim=1, name="target")
    if len(t) != len(a):
        raise ValueError("target/feature row mismatch")
    coefficient, residuals, rank, singular = np.linalg.lstsq(a, t, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    return {
        "coefficient": coefficient,
        "rank": int(rank),
        "columns": int(a.shape[1]),
        "condition": condition,
        "singular_values": singular,
        "rank_status": "FULL_COLUMN_RANK" if rank == a.shape[1] else "RANK_DEFICIENT",
        "solver": "numpy.linalg.lstsq; FP64; rcond=None; intercept first",
    }


def qr_reference(features, targets):
    """Independent reduced-QR + triangular solve reference, full rank only.

    A matrix target uses one QR factorization and one solve call, while
    representing several independently specified target fits.
    """
    a = design(features)
    t = finite_array(targets, name="reference targets")
    if t.ndim not in (1, 2) or len(t) != len(a):
        raise ValueError("reference target/feature shape mismatch")
    q, r = np.linalg.qr(a, mode="reduced")
    if r.shape != (a.shape[1], a.shape[1]):
        raise ValueError("QR reference requires n >= feature columns including intercept")
    # A separate SVD is a rank diagnostic, never a normal-equation solver.
    singular = np.linalg.svd(r, compute_uv=False)
    threshold = np.finfo(np.float64).eps * max(a.shape) * singular[0]
    if np.count_nonzero(singular > threshold) != a.shape[1]:
        raise ValueError("QR reference requires full column rank")
    return np.linalg.solve(r, q.T @ t)


def predict(features, coefficient):
    a = design(features)
    c = finite_array(coefficient, name="coefficient")
    if c.ndim not in (1, 2) or c.shape[0] != a.shape[1]:
        raise ValueError("coefficient/design shape mismatch")
    return a @ c


def mse(prediction, target):
    p = finite_array(prediction, ndim=1, name="prediction")
    t = finite_array(target, ndim=1, name="metric target")
    if p.shape != t.shape or p.size == 0:
        raise ValueError("metric row mismatch or empty metric")
    return float(np.mean(np.square(p - t), dtype=np.float64))


def quadratic_mse(features, coefficient, target):
    """Independent augmented quadratic expansion of population-on-fixed-rows MSE.

    This path does not form the prediction or the residual vector. A.T @ A is
    used solely to evaluate an already fitted coefficient, never to solve OLS.
    """
    a = design(features)
    c = finite_array(coefficient, ndim=1, name="coefficient")
    t = finite_array(target, ndim=1, name="quadratic target")
    if len(t) != len(a) or len(c) != a.shape[1]:
        raise ValueError("quadratic metric shape mismatch")
    return float((c @ (a.T @ a) @ c - 2.0 * c @ (a.T @ t) + t @ t) / len(t))


def risk_decomposition(gold_prediction, corrupted_prediction, truth):
    f0 = finite_array(gold_prediction, ndim=1)
    fs = finite_array(corrupted_prediction, ndim=1)
    y = finite_array(truth, ndim=1)
    if f0.shape != fs.shape or fs.shape != y.shape:
        raise ValueError("risk decomposition row mismatch")
    a = f0 - y
    d = fs - f0
    return {
        "risk_difference": mse(fs, y) - mse(f0, y),
        "prediction_change_energy": float(np.mean(d * d)),
        "signed_cross_term": float(2.0 * np.mean(a * d)),
    }


def transfer_ratio(delta, error_energy):
    """Ratio of energies, with no nominal denominator, epsilon, or clipping."""
    if not np.isfinite(error_energy) or error_energy == 0 or not np.isfinite(delta):
        return None, "UNDEFINED"
    return float(delta / error_energy), "DEFINED"
