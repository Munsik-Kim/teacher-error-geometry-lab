"""One dataset generation and four single-target OLS fits per replicate."""
from collections import Counter
import csv
from pathlib import Path
import resource
import time
import numpy as np
from ..artifacts import sha256, utc_now, write_json
from .model import METHODS, assigned, generate, hashes, seed, validate
from .provenance import freeze
from .risk import decomposition, fitted_risk
from .solver import counters, fit_ols
from .theory import expectation, projection

HASH_FIELDS=["x_sha256","z_sha256","noise_sha256","y_sha256","t_sha256","v_sha256","targets_sha256"]
DATA_FIELDS=["dataset_key","replicate","seed_X","seed_E","seed_H",*HASH_FIELDS,"status","error"]
COMPONENTS=("representation_loss","teacher_bias","cross_term","estimation_quadratic")
RAW_FIELDS=["row_key","replicate","method","dataset_key","x_sha256","noise_sha256","design_sha256","target_sha256",
            "coefficient_count",*[f"coef{i}" for i in range(5)],"rank","status","error","truth_risk",*COMPONENTS,"decomposition_total"]


def execute(config,output,baseline_path=None,progress=True):
    validate(config)
    run=Path(output).resolve(); run.mkdir(parents=True,exist_ok=False); (run/"raw").mkdir()
    wall=time.perf_counter(); cpu=time.process_time(); started=utc_now()
    frozen=freeze(run,config,started,baseline_path)
    g=projection(config); theory=expectation(config)
    write_json(run/"provenance/analytic_expectation.json",theory)
    before=counters(); actual=Counter(); statuses=Counter(); execution_status="IN_PROGRESS"; reason=None
    R=config["repetitions"]

    def receipt():
        after=counters()
        calls={k:after.get(k,0)-before.get(k,0) for k in ("fit_requests","fit_objects","lstsq_calls")}
        result={"stage":"P2","experiment":config["experiment"],"execution_status":execution_status,"stop_reason":reason,
                "started_at_utc":started,"last_updated_utc":utc_now(),"wall_seconds":time.perf_counter()-wall,
                "cpu_seconds":time.process_time()-cpu,"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                "planned":{"datasets":R,"fitted_rows":4*R,"solver_calls":4*R},"actual":dict(actual)|calls,
                "numerical_status_counts":dict(statuses),"source_bundle_sha256":frozen["source_bundle_sha256"],
                "config_sha256":frozen["config_sha256"],"batch_size":config["batch_size"],
                "GPU_jobs":0,"model_requests":0,"model_downloads":0,"paid_API_requests":0,"package_installs_or_upgrades":0,
                "remote_publication":0,"additional_QR_solves":0,"additional_test_datasets":0,
                "risk_definition":"population squared error against X beta, excludes observation noise variance"}
        write_json(run/"provenance/execution.json",result)
        return result

    receipt()
    try:
        with (run/"raw/datasets.csv").open("w",newline="",encoding="utf-8") as df, (run/"raw/fits.csv").open("w",newline="",encoding="utf-8") as ff:
            dw=csv.DictWriter(df,fieldnames=DATA_FIELDS); dw.writeheader()
            fw=csv.DictWriter(ff,fieldnames=RAW_FIELDS); fw.writeheader()
            for start in range(0,R,config["batch_size"]):
                if time.perf_counter()-wall>=config["wall_limit_seconds"]:
                    execution_status,reason="PARTIAL","wall cap reached at batch boundary"; break
                for rep in range(start,min(start+config["batch_size"],R)):
                    key=f"p2_s{config['rng']['streams']['X']}_rep{rep}"
                    ds={"dataset_key":key,"replicate":rep,"seed_X":str(seed(config,"X",rep)),"seed_E":str(seed(config,"E",rep)),
                        "seed_H":str(seed(config,"H",rep)),"status":"ok","error":""}|{k:"" for k in HASH_FIELDS}
                    actual["dataset_generation_attempts"]+=1
                    try:
                        data=generate(config,rep)
                        if not all(np.isfinite(v).all() for v in data.values()):
                            raise ValueError("Nonfinite generated array")
                        ds.update(hashes(data)); actual["datasets_generated"]+=1
                    except Exception as exc:
                        ds.update(status="generator_exception",error=repr(exc))
                    dw.writerow(ds)
                    for method in METHODS:
                        width=config["d"] if method in ("FO","FT") else config["r"]
                        row={k:"" for k in RAW_FIELDS}
                        row.update(row_key=f"rep{rep}:{method}",replicate=rep,method=method,dataset_key=key,
                                   x_sha256=ds["x_sha256"],noise_sha256=ds["noise_sha256"],coefficient_count=width,
                                   rank=-1,status=ds["status"],error=ds["error"])
                        for i in range(width): row[f"coef{i}"]=float("nan")
                        for name in ("truth_risk",*COMPONENTS,"decomposition_total"):row[name]=float("nan")
                        if ds["status"]=="ok":
                            design,target=assigned(data,method)
                            row["design_sha256"]=ds["x_sha256"] if method in ("FO","FT") else ds["z_sha256"]
                            row["target_sha256"]=ds[{"FO":"y_sha256","FT":"t_sha256","RO":"y_sha256","RM":"v_sha256"}[method]]
                            fit=fit_ols(design,target); row.update(rank=fit.rank,status=fit.status,error=fit.error)
                            for i,value in enumerate(fit.coefficient):row[f"coef{i}"]=float(value)
                            if fit.status=="ok":
                                try:
                                    row["truth_risk"]=fitted_risk(method,fit.coefficient,config)
                                    row.update(decomposition(method,fit.coefficient,config,g))
                                    if not all(np.isfinite(row[k]) for k in ("truth_risk",*COMPONENTS,"decomposition_total")):
                                        row.update(status="nonfinite_risk",error="Nonfinite risk/component")
                                except Exception as exc:
                                    row.update(status="risk_exception",error=repr(exc))
                        fw.writerow(row); actual["raw_rows"]+=1; actual[f"{method}_rows"]+=1; statuses[row["status"]]+=1
                df.flush(); ff.flush(); receipt()
                if progress and (start==0 or start+config["batch_size"]>=R or (start//config["batch_size"])%25==0):
                    print(f"P2 datasets {actual['dataset_generation_attempts']}/{R}, rows {actual['raw_rows']}/{4*R}",flush=True)
    except (MemoryError,OSError,KeyboardInterrupt) as exc:
        execution_status,reason="PARTIAL",f"resource/interruption {type(exc).__name__}: {exc}"
    if execution_status=="IN_PROGRESS":
        execution_status="COMPLETE" if actual["raw_rows"]==4*R else "PARTIAL"
    result=receipt()
    write_json(run/"provenance/raw_manifest.json",[{"path":"raw/"+p.name,"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted((run/"raw").glob("*.csv"))])
    return result
