"""Independent QR and fresh test-input checks, outside production fitting."""
import copy
import numpy as np
from .model import METHODS, assigned, generate, standard
from .risk import embedded, fitted_risk
from .solver import COUNTS, counters, fit_ols


def qr_solve(design,target):
    COUNTS["qr_validation_solves"]+=1
    q,r=np.linalg.qr(np.asarray(design,dtype=np.float64),mode="reduced")
    return np.linalg.solve(r,q.T@target)


def independent_mc(config):
    rows=[]; before=counters(); training=copy.deepcopy(config)
    for name in training["rng"]["streams"]:
        training["rng"]["streams"][name]+=config["rng"]["unit_fixture_stream_offset"]
    settings=config["independent_test_mc"]; N=settings["test_rows_per_fit"]
    for index,rep in enumerate(settings["training_replicates"]):
        data=generate(training,rep)
        # Test stream 3 is disjoint from all training X/E/H streams and P1.
        test=standard(config,"test_mc",index,(N,config["d"]))@np.linalg.cholesky(np.asarray(config["Sigma"],dtype=np.float64)).T
        truth=test@np.asarray(config["beta"],dtype=np.float64)
        for method in METHODS:
            design,target=assigned(data,method); fit=fit_ols(design,target)
            prediction=test@embedded(method,fit.coefficient,config)
            errors=(prediction-truth)**2; mean=float(errors.mean()); se=float(errors.std(ddof=1)/np.sqrt(N))
            exact=fitted_risk(method,fit.coefficient,config); difference=mean-exact
            passed=abs(difference)<=settings["threshold_se"]*se if se>0 else np.isclose(mean,exact,**config["tolerance"])
            rows.append({"model_set":index,"training_replicate":rep,"method":method,"test_N":N,"exact_fitted_risk":exact,
                         "test_mean":mean,"test_mc_se":se,"difference":difference,"z":difference/se if se>0 else None,"pass":bool(passed)})
    after=counters()
    return {"status":"PASS" if all(r["pass"] for r in rows) else "FAIL","checks":rows,
            "training_datasets":len(settings["training_replicates"]),"test_input_datasets":len(settings["training_replicates"]),
            "solver_calls":after.get("lstsq_calls",0)-before.get("lstsq_calls",0),
            "dependence":"four metrics share each test input set; no eight-independent-sample claim",
            "training_stream_offset":config["rng"]["unit_fixture_stream_offset"],"test_stream":config["rng"]["streams"]["test_mc"]}
