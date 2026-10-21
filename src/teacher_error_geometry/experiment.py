"""Serial bounded execution; every fit calls lstsq on one target."""
from collections import Counter
import csv
from pathlib import Path
import resource
import time
import numpy as np
from .artifacts import freeze, sha256, utc_now, write_json
from .design import validate_config
from .model import data_hash, data_seed, generate, teacher
from .risk import fitted_risks
from .solver import fit_student
from .theory import geometry

RAW_FIELDS = ["row_key", "config_id", "block", "teacher", "rho", "n", "c", "theta",
              "replicate", "dataset_key", "delta1", "delta2", "a", "b", "teacher_mse", "eta",
              "w", "truth_risk", "teacher_risk", "rank", "status", "error"]
DATA_FIELDS = ["dataset_key", "data_block", "rho_index", "n_index", "replicate", "rho", "n",
               "seed_root", "seed_stream", "seed_block", "seed_rho", "seed_n", "seed_replicate",
               "data_sha256", "status", "error"]


def execute(config, output, progress=True):
    cells, groups = validate_config(config)
    run = Path(output).resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "raw").mkdir()
    start, cpu_start = time.perf_counter(), time.process_time()
    started_at = utc_now()
    frozen = freeze(run, config, started_at)
    write_json(run / "provenance/cells.json", cells)
    planned = {"cells": len(cells), "fit_objects": len(cells) * config["repetitions"],
               "unique_training_datasets": len(groups) * config["repetitions"]}
    counts = Counter()
    statuses = Counter()
    raw_files = []
    status, reason = "IN_PROGRESS", None
    manifest_path = run / "provenance/execution.json"

    def snapshot():
        result = {"execution_status": status, "stop_reason": reason, "started_at_utc": started_at,
                  "last_updated_utc": utc_now(), "wall_seconds": time.perf_counter() - start,
                  "cpu_seconds": time.process_time() - cpu_start,
                  "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "planned": planned, "actual": dict(counts), "numerical_status_counts": dict(statuses),
                  "source_bundle_sha256": frozen["source_bundle_sha256"],
                  "config_sha256": frozen["config_sha256"],
                  "raw_files": raw_files, "gpu_jobs": 0, "model_requests": 0, "model_downloads": 0,
                  "paid_api_requests": 0, "remote_publications": 0,
                  "parallel_processes": 0, "bootstrap_executed": False, "teacher_noise_rng_used": False}
        write_json(manifest_path, result)
        return result

    snapshot()
    try:
        with (run / "raw/datasets.csv").open("w", newline="", encoding="utf-8") as dsfile:
            dsw = csv.DictWriter(dsfile, fieldnames=DATA_FIELDS)
            dsw.writeheader()
            for group in groups:
                relative = f"raw/{group['group_id']}.csv"
                raw_files.append(relative)
                with (run / relative).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
                    writer.writeheader()
                    for batch_start in range(0, config["repetitions"], config["batch_size"]):
                        if time.perf_counter() - start >= config["wall_limit_seconds"]:
                            status, reason = "PARTIAL", "wall_limit_seconds reached at batch boundary"
                            break
                        for rep in range(batch_start, min(batch_start + config["batch_size"], config["repetitions"])):
                            seed = data_seed(config["root_seed"], group["data_block"], group["rho_index"], group["n_index"], rep)
                            dataset_key = f"{group['group_id']}_rep{rep}"
                            ds = {k: group[k] for k in ("data_block", "rho_index", "n_index", "rho", "n")}
                            ds.update(dict(zip(DATA_FIELDS[7:13], seed)))
                            ds.update(dataset_key=dataset_key, replicate=rep, data_sha256="", status="ok", error="")
                            counts["dataset_generation_attempts"] += 1
                            try:
                                x1, u, x2 = generate(group["n"], group["rho"], seed)
                                ds["data_sha256"] = data_hash(x1, u, x2)
                                if not all(np.isfinite(x).all() for x in (x1, u, x2)):
                                    raise ValueError("Nonfinite generated data")
                                counts["unique_training_datasets"] += 1
                            except Exception as exc:
                                ds.update(status="generator_exception", error=repr(exc))
                            dsw.writerow(ds)
                            for cell in group["cells"]:
                                delta = [cell["delta1"], cell["delta2"]]
                                row = {k: cell[k] for k in RAW_FIELDS if k in cell}
                                row.update(geometry(cell["rho"], delta))
                                row.update(row_key=f"{cell['config_id']}:rep{rep}", replicate=rep, dataset_key=dataset_key,
                                           w=float("nan"), truth_risk=float("nan"), teacher_risk=float("nan"), rank=-1,
                                           status=ds["status"], error=ds["error"])
                                if cell["block"] == "B":
                                    counts["B_reused_dataset_references"] += 1
                                if ds["status"] == "ok":
                                    try:
                                        target = teacher(x1, x2, delta)
                                        counts["fit_requests"] += 1
                                        fit = fit_student(x1, target)
                                        counts["fit_objects"] += 1
                                        counts["solver_calls"] += int(fit.solver_called)
                                        row.update(w=fit.w, rank=fit.rank, status=fit.status, error=fit.error)
                                        if fit.status == "ok":
                                            row.update(fitted_risks(fit.w, cell["rho"], delta))
                                            if not all(np.isfinite(row[k]) for k in ("w", "truth_risk", "teacher_risk")):
                                                row.update(status="nonfinite_risk", error="Nonfinite evaluation risk")
                                    except Exception as exc:
                                        row.update(status="evaluation_exception", error=repr(exc))
                                writer.writerow(row)
                                counts["raw_rows"] += 1
                                counts[f"{cell['block']}_raw_rows"] += 1
                                statuses[row["status"]] += 1
                        f.flush()
                        dsfile.flush()
                        snapshot()
                    if status == "PARTIAL":
                        break
                if progress:
                    print(f"{group['group_id']}: {counts['raw_rows']}/{planned['fit_objects']} rows; {time.perf_counter()-start:.2f}s", flush=True)
                if status == "PARTIAL":
                    break
    except (MemoryError, OSError, KeyboardInterrupt) as exc:
        status, reason = "PARTIAL", f"resource/interruption: {type(exc).__name__}: {exc}"
    if status == "IN_PROGRESS":
        status = "COMPLETE" if counts["raw_rows"] == planned["fit_objects"] else "PARTIAL"
        if status == "PARTIAL":
            reason = "row denominator mismatch"
    result = snapshot()
    files = [run / p for p in raw_files] + [run / "raw/datasets.csv"]
    write_json(run / "provenance/raw_manifest.json", [
        {"path": p.relative_to(run).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files])
    return result
