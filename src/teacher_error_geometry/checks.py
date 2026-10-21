"""P0-only fresh test Monte Carlo; never used for full-sweep population risks."""
import numpy as np
from .model import covariance, data_seed, generate, teacher, teacher_delta
from .risk import fitted_risks
from .solver import fit_student


def independent_mc_checks(root_seed=2026102101, size=100000):
    rows = []
    for index, (rho, kind) in enumerate(((-.5, "axis_visible"), (0.0, "axis_hidden"), (.75, "projection_zero"))):
        delta = teacher_delta(kind, rho, .5)
        train_seed = data_seed(root_seed, 2, index, 0, 0, stream_id=0)
        x1, _, x2 = generate(30, rho, train_seed)
        fit = fit_student(x1, teacher(x1, x2, delta))
        test_seed = data_seed(root_seed, 2, index, 0, 0, stream_id=1)
        tx1, _, tx2 = generate(size, rho, test_seed)
        predicted = fit.w * tx1
        values = {"truth_risk": (predicted-tx1)**2,
                  "teacher_risk": (predicted-teacher(tx1, tx2, delta))**2}
        exact = fitted_risks(fit.w, rho, delta)
        for metric, data in values.items():
            mean, se = float(data.mean()), float(data.std(ddof=1)/np.sqrt(size))
            difference = mean-exact[metric]
            deterministic_zero = abs(exact[metric]) < 1e-25
            passed = abs(difference) < 1e-10 if deterministic_zero else abs(difference) <= 6*se
            rows.append({"model_index": index, "rho": rho, "teacher": kind, "w": fit.w,
                         "metric": metric, "N_test": size, "exact_fitted_risk": exact[metric],
                         "test_mean": mean, "test_mc_se": se, "difference": difference,
                         "standardized_difference": difference/se if se > 0 and not deterministic_zero else None,
                         "pass": bool(passed), "train_seed": train_seed, "test_seed": test_seed})
    # Distribution diagnostic only: the generator definition establishes covariance.
    x1, _, x2 = generate(size, .5, data_seed(root_seed, 2, 9, 0, 0, stream_id=1))
    return {"checks": rows, "passed": all(r["pass"] for r in rows),
            "covariance_smoke_diagnostic": {"N": size, "rho": .5,
                "target": covariance(.5).tolist(), "sample_covariance_ddof1": np.cov([x1,x2]).tolist(),
                "role": "diagnostic only, no sample covariance adjustment or correctness gate"},
            "fresh_test_datasets": 4, "P0_solver_calls": 3, "stream_bootstrap_used": False}
