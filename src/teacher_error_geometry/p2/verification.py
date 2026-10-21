"""Raw completeness and identities; twelve preselected dataset replays only."""
import ast
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time
import resource
import numpy as np
from ..artifacts import ROOT, read_csv, read_json, sha256, write_json
from .checks import qr_solve
from .experiment import COMPONENTS,HASH_FIELDS
from .model import METHODS, assigned, generate, hashes, seed, validate
from .provenance import preservation
from .risk import coordinate_risk,decomposition,fitted_risk
from .solver import counters,fit_ols
from .theory import projection


def verify(run,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    wall=time.perf_counter(); cpu=time.process_time()
    run=Path(run).resolve(); config=read_json(run/"provenance/config.json"); validate(config)
    R=config["repetitions"]; tol=config["tolerance"]; g=projection(config)
    frozen=read_json(run/"provenance/freeze.json"); execution=read_json(run/"provenance/execution.json")
    checks=Counter(); issues=[]
    def require(condition,label):
        checks["checks"]+=1
        if not condition:
            checks["failed_checks"]+=1
            if len(issues)<100:issues.append(label)
    require(sha256(run/"provenance/config.json")==frozen["config_sha256"],"frozen config")
    canonical=json.dumps(frozen["source_manifest"],sort_keys=True,separators=(",",":")).encode()
    require(hashlib.sha256(canonical).hexdigest()==frozen["source_bundle_sha256"],"source manifest digest")
    for item in frozen["source_manifest"]:
        for base,label in ((ROOT,"current"),(run/"provenance/source","snapshot")):
            p=base/item["path"]
            require(p.exists() and p.stat().st_size==item["bytes"] and sha256(p)==item["sha256"],f"{label} {item['path']}")
    raw_manifest=read_json(run/"provenance/raw_manifest.json")
    require({v["path"] for v in raw_manifest}=={"raw/fits.csv","raw/datasets.csv"},"raw manifest file set")
    for item in raw_manifest:
        p=run/item["path"]
        require(p.exists() and p.stat().st_size==item["bytes"] and sha256(p)==item["sha256"],f"raw hash {item['path']}")
    datasets={}; ds_reps=set()
    for row in read_csv(run/"raw/datasets.csv"):
        rep=int(row["replicate"]); key=row["dataset_key"]
        require(key not in datasets and rep not in ds_reps,f"unique dataset {key}")
        require(0<=rep<R and key==f"p2_s{config['rng']['streams']['X']}_rep{rep}",f"dataset key/range {key}")
        for stream in ("X","E","H"):
            require(ast.literal_eval(row[f"seed_{stream}"])==seed(config,stream,rep),f"seed {key} {stream}")
        datasets[key]=row; ds_reps.add(rep)
    seen=set(); per_method=defaultdict(set); status=Counter(); replays=defaultdict(list); counts=Counter()
    selected=[r for r in config["replay_replicates"] if r<R]
    require(np.array_equal(np.asarray(config["Sigma"]),np.eye(config["d"])) and
            np.array_equal(np.asarray(config["A"]),np.eye(config["d"])[:config["r"]]),"coordinate-check anchor assumptions")
    for row in read_csv(run/"raw/fits.csv"):
        counts["rows"]+=1; rep=int(row["replicate"]); m=row["method"]; key=row["dataset_key"]
        require(row["row_key"] not in seen and row["row_key"]==f"rep{rep}:{m}",f"unique row {row['row_key']}")
        seen.add(row["row_key"]); require(m in METHODS and key in datasets,f"known row references {row['row_key']}")
        if m not in METHODS or key not in datasets:continue
        ds=datasets[key]; width=config["d"] if m in ("FO","FT") else config["r"]
        require(rep==int(ds["replicate"]) and 0<=rep<R,f"paired replicate {row['row_key']}")
        require(int(row["coefficient_count"])==width and all(row[f"coef{i}"]=="" for i in range(width,5)),f"coefficient shape {row['row_key']}")
        design_hash=ds["x_sha256"] if m in ("FO","FT") else ds["z_sha256"]
        target_hash=ds[{"FO":"y_sha256","FT":"t_sha256","RO":"y_sha256","RM":"v_sha256"}[m]]
        require(row["x_sha256"]==ds["x_sha256"] and row["noise_sha256"]==ds["noise_sha256"] and
                row["design_sha256"]==design_hash and row["target_sha256"]==target_hash,f"data/target pairing {row['row_key']}")
        per_method[m].add(rep); status[row["status"]]+=1
        if row["status"]=="ok":
            w=np.array([float(row[f"coef{i}"]) for i in range(width)],dtype=np.float64)
            risk=float(row["truth_risk"])
            require(int(row["rank"])==width and np.isfinite(w).all() and np.isfinite(risk),f"finite/rank {row['row_key']}")
            require(np.isclose(risk,fitted_risk(m,w,config),**tol) and np.isclose(risk,coordinate_risk(m,w,config),**tol),f"quadratic/coordinate {row['row_key']}")
            components=decomposition(m,w,config,g)
            require(all(np.isclose(float(row[k]),components[k],**tol) for k in (*COMPONENTS,"decomposition_total")),f"persisted components {row['row_key']}")
            require(np.isclose(risk,sum(float(row[k]) for k in COMPONENTS),**tol),f"sample decomposition identity {row['row_key']}")
            if m=="RM":
                beta=np.array(config["beta"]); delta=np.array(config["delta"]); l=config["lambda"]; r=config["r"]
                h=w-beta[:r]-l*delta[:r]
                independent_cross=sum(2*l*delta[i]*h[i] for i in range(r))
                expanded=sum(beta[r:]**2)+l*l*sum(delta[:r]**2)+independent_cross+sum(h*h)
                require(np.isclose(risk,expanded,**tol) and np.isclose(float(row["cross_term"]),independent_cross,**tol),f"independent RM signed cross {row['row_key']}")
        if rep in selected:replays[rep].append(row)
    require(counts["rows"]==execution["actual"].get("raw_rows",0),"actual raw denominator")
    require(len(datasets)==execution["actual"].get("dataset_generation_attempts",0),"actual dataset denominator")
    if execution["execution_status"]=="COMPLETE":
        require(counts["rows"]==4*R and len(datasets)==R,"planned complete denominator")
        require(all(per_method[m]==set(range(R)) for m in METHODS),"four paired method replicate coverage")
    before=counters(); replay_rows=0
    for rep in selected:
        if rep not in replays:continue
        data=generate(config,rep); actual_hashes=hashes(data); fits={}
        key=replays[rep][0]["dataset_key"]; ds=datasets[key]
        require(all(actual_hashes[k]==ds[k] for k in HASH_FIELDS),f"seed replay all hashes rep{rep}")
        for row in replays[rep]:
            m=row["method"]; design,target=assigned(data,m); fit=fit_ols(design,target); fits[m]=fit; replay_rows+=1
            saved=[float(row[f"coef{i}"]) for i in range(int(row["coefficient_count"]))]
            require(fit.status==row["status"] and np.allclose(fit.coefficient,saved,**tol),f"lstsq coefficient replay rep{rep}:{m}")
            if fit.status=="ok":
                require(np.allclose(fit.coefficient,qr_solve(design,target),**tol),f"independent QR rep{rep}:{m}")
                require(np.isclose(fitted_risk(m,fit.coefficient,config),float(row["truth_risk"]),**tol),f"risk replay rep{rep}:{m}")
        teacher_head=fit_ols(data["Z"],data["T"])
        if "RO" in fits and "RM" in fits:
            combined=(1-config["lambda"])*fits["RO"].coefficient+config["lambda"]*teacher_head.coefficient
            require(np.allclose(fits["RM"].coefficient,combined,**tol),f"mixed-target linearity rep{rep}")
    after=counters(); extra={k:after.get(k,0)-before.get(k,0) for k in ("fit_requests","fit_objects","lstsq_calls","qr_validation_solves")}
    p1=None
    if (run/"provenance/p1_before.json").exists():
        p1=preservation(run/"provenance/p1_before.json"); require(p1["status"]=="PASS","P1 outputs/source unchanged")
    result={"implementation_verification":"PASS" if not checks["failed_checks"] else "FAIL",
            "execution_status":execution["execution_status"],"checks":checks["checks"],"failed_checks":checks["failed_checks"],
            "issues_first_100":issues,"raw_rows_checked":counts["rows"],"unique_dataset_metadata_checked":len(datasets),
            "numerical_status_counts":dict(status),"replayed_replicates":selected,"replayed_method_rows":replay_rows,
            "verification_extra_calls":extra,"verification_regenerated_datasets":len(selected),
            "wall_seconds":time.perf_counter()-wall,"cpu_seconds":time.process_time()-cpu,
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "P1_preservation":p1,"full_sweep_repeated":False,"AI_assisted_internal_check":True,"human_external_review":False}
    write_json(output,result)
    return result
