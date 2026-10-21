"""Fitted population risk and sample decomposition; never substitute an expectation."""
import numpy as np
from .model import arrays


def embedded(method,coefficient,config):
    w=np.asarray(coefficient,dtype=np.float64)
    return w if method in ("FO","FT") else np.asarray(config["A"],dtype=np.float64).T@w


def fitted_risk(method,coefficient,config):
    theta=embedded(method,coefficient,config)
    p=arrays(config); error=theta-p["beta"]
    if theta.shape!=(config["d"],) or not np.isfinite(theta).all():
        raise ValueError("Invalid fitted coefficient")
    return float(error@p["Sigma"]@error)


def coordinate_risk(method,coefficient,config):
    """Independent anchor-only coordinate check, no embedding/expectation calls."""
    beta=np.asarray(config["beta"],dtype=np.float64); w=np.asarray(coefficient,dtype=np.float64)
    if method in ("FO","FT"):
        return float(np.sum((w-beta)**2))
    r=config["r"]
    return float(sum(beta[i]**2 for i in range(r,config["d"]))+sum((w[i]-beta[i])**2 for i in range(r)))


def decomposition(method,coefficient,config,g):
    p=arrays(config); w=np.asarray(coefficient,dtype=np.float64); l=config["lambda"]
    representation=bias=cross=0.
    if method=="FO":
        h=w-p["beta"]; quadratic=float(h@p["Sigma"]@h)
    elif method=="FT":
        h=w-(p["beta"]+p["delta"])
        bias=g["B"]; cross=float(2*p["delta"]@p["Sigma"]@h); quadratic=float(h@p["Sigma"]@h)
    else:
        representation=g["B_A"]
        center=g["w_beta"] if method=="RO" else (1-l)*g["w_beta"]+l*g["w_gamma"]
        h=w-center; quadratic=float(h@g["C"]@h)
        if method=="RM":
            bias=l*l*g["D_A"]; cross=float(2*l*(g["w_gamma"]-g["w_beta"])@g["C"]@h)
    return {"representation_loss":representation,"teacher_bias":bias,"cross_term":cross,
            "estimation_quadratic":quadratic,"decomposition_total":representation+bias+cross+quadratic}
