"""Persisted raw -> four marginal and six paired comparisons, plus components."""
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
from ..artifacts import read_csv, read_json, write_csv, write_json
from .experiment import COMPONENTS
from .model import METHODS
from .theory import expectation, projection


def identical_targets(config):
    return all(v==0 for v in config["delta"]) and config["sigma2"]==config["tau2"]==config["omega"]


def method_deterministic(method,config):
    g=projection(config)
    if method=="FO":return config["sigma2"]==0
    if method=="FT":return config["tau2"]==0
    if method=="RO":return g["B_A"]+config["sigma2"]==0
    return expectation(config)["rm_bracket"]==0


def pair_deterministic(left,right,config):
    if method_deterministic(left,config) and method_deterministic(right,config):return True
    def target(m):
        if identical_targets(config):return "Y"
        if m in ("FO","RO"):return "Y"
        if m=="FT":return "T"
        return "Y" if config["lambda"]==0 else "T" if config["lambda"]==1 else "V"
    same_space=(left in ("FO","FT"))==(right in ("FO","FT")) or config["r"]==config["d"]
    return same_space and target(left)==target(right)


def stats(values,theory,config,deterministic=False):
    x=np.asarray(values,dtype=np.float64)
    if not x.size:
        return {k:None for k in ("mean","sd","mc_se","ci95_low","ci95_high","difference","z")} | {"comparison":"NO_VALID_ROWS"}
    mean=float(x.mean()); sd=float(x.std(ddof=1)) if x.size>1 else None
    se=sd/math.sqrt(x.size) if sd is not None else None; diff=mean-theory; z=None
    if deterministic:
        comparison="DETERMINISTIC_PASS" if np.allclose(x,theory,**config["tolerance"]) else "DETERMINISTIC_FAIL"
    elif se is None or se==0:
        comparison="ZERO_SE_INVESTIGATE"
    else:
        z=diff/se; comparison="INVESTIGATE_4SE" if abs(z)>config["diagnostics"]["z_threshold"] else "WITHIN_4SE"
    hw=config["diagnostics"]["pointwise_z"]*se if se is not None else None
    return {"mean":mean,"sd":sd,"mc_se":se,"ci95_low":mean-hw if hw is not None else None,
            "ci95_high":mean+hw if hw is not None else None,"difference":diff,"z":z,"comparison":comparison}


def summarize(run,output):
    run=Path(run).resolve(); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    config=read_json(run/"provenance/config.json"); expected=expectation(config); g=projection(config); R=config["repetitions"]
    raw=defaultdict(list); failures=[]
    fields=("truth_risk",*COMPONENTS,"decomposition_total")
    for row in read_csv(run/"raw/fits.csv"):
        item={k:row[k] for k in ("row_key","method","dataset_key","status","error")}
        item["replicate"]=int(row["replicate"])
        item.update({k:float(row[k]) for k in fields})
        raw[row["method"]].append(item)
        if row["status"]!="ok":failures.append({k:row[k] for k in ("row_key","method","dataset_key","status","error")})
    marginal=[]; component_rows=[]; paired=[]
    for method in METHODS:
        observations=sorted(raw[method],key=lambda r:r["replicate"])
        valid=[o for o in observations if o["status"]=="ok" and all(np.isfinite(o[k]) for k in fields)]
        base={"method":method,"R_planned":R,"R_rows":len(observations),"N_valid":len(valid),
              "N_failed":len(observations)-len(valid),"N_missing":R-len(observations)}
        marginal.append(base|{"theory":expected["risks"][method],"deterministic":method_deterministic(method,config)}|
                        stats([o["truth_risk"] for o in valid],expected["risks"][method],config,method_deterministic(method,config)))
        for component in (*COMPONENTS,"decomposition_total"):
            theory=expected["risks"][method] if component=="decomposition_total" else expected["components"][method][component]
            deterministic=component in ("representation_loss","teacher_bias") or (component=="cross_term" and
                (method in ("FO","RO") or method=="FT" and g["B"]==0 or method=="RM" and (config["lambda"]==0 or g["D_A"]==0)))
            if component in ("decomposition_total","estimation_quadratic"):
                deterministic=method_deterministic(method,config)
            component_rows.append(base|{"component":component,"theory":theory,"deterministic":deterministic}|
                                  stats([o[component] for o in valid],theory,config,deterministic))
    indexed={m:{o["replicate"]:o for o in raw[m]} for m in METHODS}
    for left,right in config["paired_contrasts"]:
        values=[]; mismatch=0
        for rep in sorted(indexed[left].keys() & indexed[right].keys()):
            l,r=indexed[left][rep],indexed[right][rep]
            if l["dataset_key"]!=r["dataset_key"]:mismatch+=1; continue
            if l["status"]==r["status"]=="ok" and np.isfinite(l["truth_risk"]) and np.isfinite(r["truth_risk"]):
                values.append(l["truth_risk"]-r["truth_risk"])
        theory=expected["risks"][left]-expected["risks"][right]
        det=pair_deterministic(left,right,config)
        paired.append({"left":left,"right":right,"sign":"left minus right","R_planned":R,"N_pairs":len(values),
                       "N_failed_or_missing_pairs":R-len(values),"dataset_key_mismatches":mismatch,"theory":theory,"deterministic":det}|
                      stats(values,theory,config,det))
    write_csv(output/"methods.csv",marginal); write_csv(output/"paired.csv",paired); write_csv(output/"decomposition.csv",component_rows)
    write_csv(output/"failures.csv",failures,["row_key","method","dataset_key","status","error"])
    flagged=lambda rows:[r for r in rows if "INVESTIGATE" in r["comparison"] or "FAIL" in r["comparison"]]
    flags={"methods":flagged(marginal),"paired":flagged(paired),"components":flagged(component_rows)}
    result={"execution_status":read_json(run/"provenance/execution.json")["execution_status"],"config":config,
            "methods":marginal,"paired":paired,"components":component_rows,"flagged":flags,
            "statistical_comparison":"INVESTIGATE" if any(flags.values()) else "NO_4SE_FLAGS",
            "raw_rows":sum(len(v) for v in raw.values()),"failed_rows":len(failures),
            "missing_rows":sum(R-len(raw[m]) for m in METHODS),
            "source_bundle_sha256":read_json(run/"provenance/freeze.json")["source_bundle_sha256"],
            "intervals":"pointwise normal mean +/-1.96 sample_SD/sqrt(N_valid), ddof=1; not simultaneous",
            "dependence":"four fits and six paired differences share each of 20000 datasets; do not sum them as independent samples"}
    write_json(output/"overview.json",result)
    return result
