"""Rebuild summaries exclusively from persisted raw rows and frozen config."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import math
import numpy as np
from .artifacts import read_csv, read_json, write_csv, write_json
from .theory import expectation, geometry, risk_variance

METRICS = ("truth_risk", "teacher_risk")


def paired_deterministic(left, right, metric):
    """Recognize exact shared-data risk identities, including angle symmetry.

    At rho=0, normalized theta and pi-theta have the same b in real
    arithmetic. Unequal last bits of sin/cos must not turn that identity into
    a Monte Carlo test. This is a structural identity, not an SE cutoff.
    """
    ld = [left["delta1"], left["delta2"]]
    rd = [right["delta1"], right["delta2"]]
    if ld == rd:
        return True
    lg, rg = geometry(left["rho"], ld), geometry(right["rho"], rd)
    if metric == "teacher_risk":
        if lg["b"]**2 == rg["b"]**2:
            return True
        return (left["teacher"] == right["teacher"] == "normalized_angle"
                and left["rho"] == right["rho"] == 0
                and left["theta"] + right["theta"] == math.pi)
    if metric == "truth_risk":
        return lg["b"]**2 == rg["b"]**2 and lg["a"]*lg["b"] == rg["a"]*rg["b"]
    raise ValueError("Unknown metric")


def collect(run):
    run = Path(run)
    groups = defaultdict(list)
    for path in sorted((run / "raw").glob("d*.csv")):
        if path.name == "datasets.csv":
            continue
        for row in read_csv(path):
            for key in ("w", *METRICS):
                row[key] = float(row[key])
            row["replicate"] = int(row["replicate"])
            groups[row["config_id"]].append(row)
    return groups


def sample_stats(values, theoretical, config, deterministic=False):
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {k: None for k in ("mean", "sd", "mc_se", "ci95_low", "ci95_high", "difference", "standardized_difference")} | {"comparison": "NO_VALID_ROWS"}
    mean = float(data.mean())
    sd = float(data.std(ddof=1)) if data.size > 1 else None
    se = sd / math.sqrt(data.size) if sd is not None else None
    difference = mean - theoretical
    tol = config["tolerance"]
    within_tolerance = abs(difference) <= tol["atol"] + tol["rtol"] * abs(theoretical)
    z = None
    if deterministic:
        comparison = "DETERMINISTIC_PASS" if np.allclose(data, theoretical, **tol) else "DETERMINISTIC_FAIL"
    elif se is None or se == 0:
        comparison = "ZERO_SE_WITHIN_TOLERANCE" if within_tolerance else "ZERO_SE_INVESTIGATE"
    else:
        z = difference / se
        comparison = "INVESTIGATE_4SE" if abs(z) > config["diagnostics"]["z_threshold"] else "WITHIN_4SE"
    halfwidth = config["diagnostics"]["pointwise_z"] * se if se is not None else None
    return {"mean": mean, "sd": sd, "mc_se": se,
            "ci95_low": mean - halfwidth if halfwidth is not None else None,
            "ci95_high": mean + halfwidth if halfwidth is not None else None,
            "difference": difference, "standardized_difference": z, "comparison": comparison}


def summarize(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(run / "provenance/config.json")
    cells = read_json(run / "provenance/cells.json")
    raw = collect(run)
    rows, paired, failures = [], [], []
    for cell in cells:
        observations = sorted(raw.get(cell["config_id"], []), key=lambda x: x["replicate"])
        delta = [cell["delta1"], cell["delta2"]]
        theory = expectation(cell["n"], cell["rho"], delta)
        geo = geometry(cell["rho"], delta)
        for observation in observations:
            if observation["status"] != "ok":
                failures.append({k: observation[k] for k in ("row_key", "dataset_key", "config_id", "status", "error")})
        for metric in METRICS:
            good = [o[metric] for o in observations if o["status"] == "ok" and np.isfinite(o[metric])]
            record = dict(cell) | geo | {"delta_euclidean_norm": float(np.linalg.norm(delta)),
                       "metric": metric, "R_planned": config["repetitions"], "R_rows": len(observations),
                       "N_valid": len(good), "N_failed": len(observations) - len(good),
                       "N_missing": config["repetitions"] - len(observations),
                       "theory": theory[metric], "deterministic": geo["b"] == 0.0,
                       "theoretical_risk_sd": math.sqrt(risk_variance(cell["n"], cell["rho"], delta, metric))}
            record.update(sample_stats(good, theory[metric], config, record["deterministic"]))
            rows.append(record)
    by_group = defaultdict(list)
    for cell in cells:
        by_group[cell["group_id"]].append(cell)
    for group in by_group.values():
        for left, right in combinations(group, 2):
            lrows = {o["replicate"]: o for o in raw.get(left["config_id"], [])}
            rrows = {o["replicate"]: o for o in raw.get(right["config_id"], [])}
            for metric in METRICS:
                diffs = []
                mismatched = 0
                for rep in sorted(lrows.keys() & rrows.keys()):
                    lrow, rrow = lrows[rep], rrows[rep]
                    if lrow["dataset_key"] != rrow["dataset_key"]:
                        mismatched += 1
                        continue
                    if lrow["status"] == rrow["status"] == "ok" and np.isfinite(lrow[metric]) and np.isfinite(rrow[metric]):
                        diffs.append(lrow[metric] - rrow[metric])
                ld, rd = [left["delta1"], left["delta2"]], [right["delta1"], right["delta2"]]
                theory_difference = expectation(left["n"], left["rho"], ld)[metric] - expectation(right["n"], right["rho"], rd)[metric]
                identical_teacher = ld == rd
                deterministic = paired_deterministic(left, right, metric)
                # A_hidden and B at rho=0 are the same target on the same data.
                item = {"left_config_id": left["config_id"], "right_config_id": right["config_id"],
                        "rho": left["rho"], "n": left["n"], "metric": metric,
                        "R_planned": config["repetitions"], "N_pairs": len(diffs),
                        "N_incomplete_or_failed": config["repetitions"] - len(diffs),
                        "dataset_key_mismatches": mismatched, "theory_difference": theory_difference,
                        "identical_teacher_reuse": identical_teacher,
                        "independent_evidence": not identical_teacher}
                item.update(sample_stats(diffs, theory_difference, config, deterministic))
                paired.append(item)
    write_csv(output / "cells.csv", rows)
    write_csv(output / "paired_differences.csv", paired)
    write_csv(output / "failures.csv", failures, ["row_key", "dataset_key", "config_id", "status", "error"])
    flagged = [r for r in rows if "INVESTIGATE" in r["comparison"] or "FAIL" in r["comparison"]]
    paired_flagged = [r for r in paired if "INVESTIGATE" in r["comparison"] or "FAIL" in r["comparison"]]
    result = {"execution_status": read_json(run / "provenance/execution.json")["execution_status"],
              "cells": len(cells), "metric_cells": len(rows), "paired_metric_comparisons": len(paired),
              "raw_rows": sum(len(v) for v in raw.values()), "failed_rows": len(failures),
              "missing_rows": sum(config["repetitions"] - len(raw.get(c["config_id"], [])) for c in cells),
              "cell_comparisons": dict(Counter(r["comparison"] for r in rows)),
              "flagged_metric_cells": flagged, "flagged_paired_comparisons": paired_flagged,
              "statistical_comparison": "INVESTIGATE" if flagged or paired_flagged else "NO_4SE_FLAGS",
              "interval_definition": "mean +/- 1.96 * sample_SD/sqrt(N_valid); pointwise normal approximation, no simultaneous coverage claim",
              "independence_note": "Do not sum reused B data, multiple risks of one fit, or paired teacher targets as independent observations",
              "config": config, "source_bundle_sha256": read_json(run / "provenance/freeze.json")["source_bundle_sha256"]}
    write_json(output / "overview.json", result)
    return result
