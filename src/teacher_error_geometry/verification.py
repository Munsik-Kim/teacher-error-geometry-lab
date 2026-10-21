"""Frozen-source, raw integrity, pairing and fixed-key numerical replay checks."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .artifacts import ROOT, read_csv, read_json, sha256, write_json
from .design import validate_config
from .model import data_hash, data_seed, generate, teacher
from .risk import conditional_teacher_risk, fitted_risks
from .solver import fit_student
from .theory import geometry


def verify(run, output=None):
    run = Path(run).resolve()
    config = read_json(run / "provenance/config.json")
    tol = config["tolerance"]
    cells, groups = validate_config(config)
    by_cell = {c["config_id"]: c for c in cells}
    by_group = {g["group_id"]: g for g in groups}
    frozen = read_json(run / "provenance/freeze.json")
    execution = read_json(run / "provenance/execution.json")
    issues = []
    counters = Counter()

    def require(condition, label):
        counters["checks"] += 1
        if not condition:
            counters["failed_checks"] += 1
            if len(issues) < 100:
                issues.append(label)

    require(sha256(run / "provenance/config.json") == frozen["config_sha256"], "frozen config hash")
    canonical = json.dumps(frozen["source_manifest"], sort_keys=True, separators=(",", ":")).encode()
    require(hashlib.sha256(canonical).hexdigest() == frozen["source_bundle_sha256"], "source bundle manifest hash")
    for source in frozen["source_manifest"]:
        relative = source["path"]
        for base, label in ((run / "provenance/source", "snapshot"), (ROOT, "current")):
            file = base / relative
            require(file.exists() and file.stat().st_size == source["bytes"] and sha256(file) == source["sha256"], f"{label} source {relative}")
    for file in read_json(run / "provenance/raw_manifest.json"):
        path = run / file["path"]
        require(path.exists() and path.stat().st_size == file["bytes"] and sha256(path) == file["sha256"], f"raw manifest {file['path']}")
    datasets = {}
    for ds in read_csv(run / "raw/datasets.csv"):
        key = ds["dataset_key"]
        require(key not in datasets, f"duplicate dataset {key}")
        seed = [int(ds[k]) for k in ("seed_root", "seed_stream", "seed_block", "seed_rho", "seed_n", "seed_replicate")]
        group_id = f"d{ds['data_block']}_r{ds['rho_index']}_n{ds['n_index']}"
        group = by_group.get(group_id)
        require(group is not None, f"unknown dataset group {key}")
        if group is not None:
            expected = data_seed(config["root_seed"], group["data_block"], group["rho_index"], group["n_index"], int(ds["replicate"]))
            require(seed == expected and float(ds["rho"]) == group["rho"] and int(ds["n"]) == group["n"]
                    and key == f"{group_id}_rep{ds['replicate']}", f"seed mapping {key}")
        datasets[key] = ds
    row_keys, reps = set(), defaultdict(set)
    pair_targets = defaultdict(dict)
    replay_rows = defaultdict(list)
    replay_keys = set()
    for index in (0, len(groups)//2, len(groups)-1):
        for rep in sorted(set([0, 1, config["repetitions"]//2, config["repetitions"]-1])):
            replay_keys.add(f"{groups[index]['group_id']}_rep{rep}")
    statuses = Counter()
    for path in sorted((run / "raw").glob("d*.csv")):
        if path.name == "datasets.csv":
            continue
        for row in read_csv(path):
            counters["raw_rows"] += 1
            cid, key, rep = row["config_id"], row["dataset_key"], int(row["replicate"])
            statuses[row["status"]] += 1
            require(row["row_key"] not in row_keys, f"duplicate raw row {row['row_key']}")
            row_keys.add(row["row_key"])
            require(cid in by_cell and key in datasets, f"unknown cell/dataset {row['row_key']}")
            if cid not in by_cell or key not in datasets:
                continue
            cell = by_cell[cid]
            require(0 <= rep < config["repetitions"] and row["row_key"] == f"{cid}:rep{rep}"
                    and key == f"{cell['group_id']}_rep{rep}", f"row seed/key pairing {row['row_key']}")
            for name in ("rho", "n", "c", "delta1", "delta2"):
                require(float(row[name]) == cell[name], f"row metadata {row['row_key']} {name}")
            require(row["block"] == cell["block"] and row["teacher"] == cell["teacher"], f"teacher metadata {row['row_key']}")
            require((row["theta"] == "" and cell["theta"] is None) or (row["theta"] != "" and float(row["theta"]) == cell["theta"]), f"theta metadata {row['row_key']}")
            reps[cid].add(rep)
            delta = [cell["delta1"], cell["delta2"]]
            geo = geometry(cell["rho"], delta)
            require(all(np.isclose(float(row[name]), geo[name], **tol) for name in ("a", "b", "teacher_mse")), f"geometry metadata {row['row_key']}")
            if row["status"] == "ok":
                w = float(row["w"])
                risks = fitted_risks(w, cell["rho"], delta)
                require(int(row["rank"]) == 1 and np.isfinite(w), f"rank/coefficient {row['row_key']}")
                require(all(np.isclose(float(row[name]), risks[name], **tol) for name in risks), f"persisted risks {row['row_key']}")
                require(np.isclose(float(row["teacher_risk"]), conditional_teacher_risk(w, cell["rho"], delta), **tol), f"quadratic/conditional risk {row['row_key']}")
                if cell["block"] in ("A", "B") and cell["rho"] == 0 and cell["n"] in config["B"]["n"] and cell["teacher"] in ("axis_hidden", "projection_zero"):
                    pair_targets[key][cell["teacher"]] = (w, float(row["truth_risk"]), float(row["teacher_risk"]))
            if key in replay_keys:
                replay_rows[key].append(row)
    require(counters["raw_rows"] == execution["actual"].get("raw_rows", 0), "execution actual raw count")
    require(len(datasets) == execution["actual"].get("dataset_generation_attempts", 0), "dataset generation count")
    if execution["execution_status"] == "COMPLETE":
        require(counters["raw_rows"] == len(cells)*config["repetitions"], "full fit denominator")
        require(len(datasets) == len(groups)*config["repetitions"], "full unique dataset denominator")
        require(all(reps[c["config_id"]] == set(range(config["repetitions"])) for c in cells), "all cell replicate coverage")
    for key, targets in pair_targets.items():
        require(len(targets) == 2 and targets.get("axis_hidden") == targets.get("projection_zero"), f"rho=0 duplicate teacher reuse {key}")
    replay_fits = 0
    for key in sorted(replay_keys):
        if key not in datasets:
            continue  # partial runs disclose missing denominators above and in summaries
        ds = datasets[key]
        seed = [int(ds[k]) for k in ("seed_root", "seed_stream", "seed_block", "seed_rho", "seed_n", "seed_replicate")]
        x1, u, x2 = generate(int(ds["n"]), float(ds["rho"]), seed)
        require(data_hash(x1, u, x2) == ds["data_sha256"], f"dataset hash replay {key}")
        for row in replay_rows[key]:
            cell = by_cell[row["config_id"]]
            delta = [cell["delta1"], cell["delta2"]]
            target = teacher(x1, x2, delta)
            fit = fit_student(x1, target)
            replay_fits += int(fit.solver_called)
            require(fit.status == row["status"] and np.isclose(fit.w, float(row["w"]), **tol), f"coefficient replay {row['row_key']}")
            scalar = float(np.sum(x1 * target) / np.sum(x1 * x1))
            require(np.isclose(fit.w, scalar, **tol), f"independent scalar solver {row['row_key']}")
            a, b = geometry(cell["rho"], delta)["a"], geometry(cell["rho"], delta)["b"]
            oracle_w = 1 + a + b * float(np.sum(x1*u) / np.sum(x1*x1))
            require(np.isclose(fit.w, oracle_w, **tol), f"coefficient algebra {row['row_key']}")
            for metric, value in fitted_risks(fit.w, cell["rho"], delta).items():
                require(np.isclose(value, float(row[metric]), **tol), f"risk replay {row['row_key']}")
    result = {"implementation_verification": "PASS" if not counters["failed_checks"] else "FAIL",
              "execution_status": execution["execution_status"], "raw_rows_checked": counters["raw_rows"],
              "unique_dataset_metadata_checked": len(datasets), "numerical_status_counts": dict(statuses),
              "checks": counters["checks"], "failed_checks": counters["failed_checks"], "issues_first_100": issues,
              "replayed_dataset_keys": sorted(replay_keys & datasets.keys()), "replay_solver_calls": replay_fits,
              "full_rerun_performed": False, "AI_assisted_internal_verification": True,
              "scope": "data/solver/risk/provenance consistency for P1 only, not independent human or external review"}
    write_json(Path(output) if output else run / "provenance/verification.json", result)
    return result
