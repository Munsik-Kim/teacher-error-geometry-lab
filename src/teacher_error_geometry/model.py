"""Data and deterministic teachers. No finite-sample covariance adjustment."""
import hashlib
import numpy as np


def covariance(rho):
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("rho must be finite and strictly between -1 and 1")
    return np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)


def data_seed(root_seed, block_id, rho_index, n_index, replicate, stream_id=0):
    values = [root_seed, stream_id, block_id, rho_index, n_index, replicate]
    if any(int(v) != v or v < 0 for v in values):
        raise ValueError("Seed components must be nonnegative integers")
    return [int(v) for v in values]


def generate(n, rho, seed):
    covariance(rho)
    if int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    x1, u = rng.standard_normal((2, int(n)), dtype=np.float64)
    x2 = rho * x1 + np.sqrt(1.0 - rho * rho) * u
    return x1, u, x2


def data_hash(x1, u, x2):
    digest = hashlib.sha256()
    for array in (x1, u, x2):
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def teacher(x1, x2, delta):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("delta must contain two finite coefficients")
    return (1.0 + delta[0]) * x1 + delta[1] * x2


def teacher_delta(kind, rho, c, theta=None):
    sigma = covariance(rho)
    if not np.isfinite(c) or c < 0:
        raise ValueError("c must be finite and nonnegative")
    if kind == "axis_visible":
        return np.array([c, 0.0], dtype=np.float64)
    if kind == "axis_hidden":
        return np.array([0.0, c], dtype=np.float64)
    if kind == "projection_zero":
        return c * np.array([-rho, 1.0]) / np.sqrt(1.0 - rho * rho)
    if kind == "normalized_angle" and theta is not None and np.isfinite(theta):
        v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        return c * v / np.sqrt(v @ sigma @ v)
    raise ValueError("Unknown teacher kind or missing/nonfinite theta")
