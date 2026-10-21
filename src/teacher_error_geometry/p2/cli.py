"""P2 CLI extension; existing P1 subcommands retain their meanings."""
import copy
import io
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest
from ..artifacts import ROOT,read_json,write_json


def register(sub):
    p2=sub.add_parser("p2",help="P2 single original-paper Gaussian anchor")
    nested=p2.add_subparsers(dest="p2_command",required=True)
    for name in ("smoke","run"):
        parser=nested.add_parser(name)
        parser.add_argument("--config",default=str(ROOT/"configs/p2_paper_anchor.json"))
        parser.add_argument("--output",required=True)
        parser.add_argument("--p1-baseline", help="Optional legacy P1 preservation inventory")
        if name=="smoke":parser.add_argument("--reuse-p1-regression-receipt")
    parser=nested.add_parser("verify"); parser.add_argument("--run",required=True); parser.add_argument("--output",required=True)
    parser=nested.add_parser("summarize"); parser.add_argument("--run",required=True); parser.add_argument("--output",required=True)
    parser=nested.add_parser("figures"); parser.add_argument("--summary",required=True); parser.add_argument("--output",required=True)


def dispatch(args):
    if args.p2_command=="run":
        from .experiment import execute
        r=execute(read_json(args.config),args.output,args.p1_baseline)
        print({k:r[k] for k in ("execution_status","wall_seconds","actual")}); return 0 if r["execution_status"]=="COMPLETE" else 2
    if args.p2_command=="verify":
        from .verification import verify
        r=verify(args.run,args.output)
        print({k:r[k] for k in ("implementation_verification","raw_rows_checked","replayed_method_rows","verification_extra_calls","wall_seconds")})
        return 0 if r["implementation_verification"]=="PASS" else 1
    if args.p2_command=="summarize":
        from .analysis import summarize
        r=summarize(args.run,args.output)
        print({k:r[k] for k in ("execution_status","raw_rows","failed_rows","missing_rows","statistical_comparison")}); return 0
    if args.p2_command=="figures":
        from .plotting import figures
        r=figures(args.summary,args.output); print({"figure_count":r["figure_count"],"files":len(r["files"])}); return 0
    if args.p2_command=="smoke":return smoke(args)


def smoke(args):
    from .analysis import summarize
    from .checks import independent_mc
    from .experiment import execute
    from .plotting import figures
    from .provenance import preservation
    from .solver import counters
    from .verification import verify
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); cpu=time.process_time(); before=counters()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests_p2"),pattern="test_*.py")
    stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (output/"p2_unittest.txt").write_text(stream.getvalue(),encoding="utf-8"); print(stream.getvalue(),flush=True)
    after=counters(); units={"status":"PASS" if result.wasSuccessful() else "FAIL","tests_run":result.testsRun,
        "wall_seconds":time.perf_counter()-start,"cpu_seconds":time.process_time()-cpu,
        "calls":{k:after.get(k,0)-before.get(k,0) for k in ("lstsq_calls","qr_validation_solves")}}
    write_json(output/"p2_unittest.json",units)
    if not result.wasSuccessful():write_json(output/"smoke.json",{"status":"FAIL","phase":"P2 unit tests"}); return 1
    p1start=time.perf_counter()
    if args.reuse_p1_regression_receipt:
        p1=read_json(args.reuse_p1_regression_receipt)
        if p1["status"]!="PASS" or p1["tests_run"]!=17:raise ValueError("Invalid P1 regression receipt")
        p1=dict(p1)|{"reused_receipt":str(Path(args.reuse_p1_regression_receipt).resolve()),"new_invocation":False}
    else:
        child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
        p=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(ROOT/"tests"),"-p","test_contract.py","-v"],cwd=ROOT,capture_output=True,text=True)
        child_end=resource.getrusage(resource.RUSAGE_CHILDREN)
        (output/"p1_regression.txt").write_text(p.stdout+p.stderr,encoding="utf-8")
        p1={"status":"PASS" if p.returncode==0 and "Ran 17 tests" in p.stderr else "FAIL","tests_run":17,
            "wall_seconds":time.perf_counter()-p1start,"cpu_seconds":child_end.ru_utime+child_end.ru_stime-child_start.ru_utime-child_start.ru_stime,
            "aggregate_solver_calls":"not instrumented by original P1 tests","new_invocation":True,"P1_full_executed":False}
    write_json(output/"p1_regression.json",p1)
    if p1["status"]!="PASS":write_json(output/"smoke.json",{"status":"FAIL","phase":"P1 regression"}); return 1
    config=read_json(args.config)
    phase=time.perf_counter(); phasecpu=time.process_time(); mc=independent_mc(config)
    mc.update(wall_seconds=time.perf_counter()-phase,cpu_seconds=time.process_time()-phasecpu)
    write_json(output/"independent_mc.json",mc)
    if mc["status"]!="PASS":write_json(output/"smoke.json",{"status":"FAIL","phase":"independent test MC"}); return 1
    pilot_config=copy.deepcopy(config); pilot_config["experiment"]="paper_anchor_pilot_NOT_FULL"; pilot_config["repetitions"]=50
    pilot_config["planned"]={"unique_training_datasets":50,"fit_objects":200,"solver_calls":200}
    for k in pilot_config["rng"]["streams"]:pilot_config["rng"]["streams"][k]+=config["rng"]["pilot_stream_offset"]
    pilot=execute(pilot_config,output/"pilot",args.p1_baseline,progress=False)
    verified=verify(output/"pilot",output/"pilot_verification.json")
    summary=summarize(output/"pilot",output/"pilot/summary"); figures(output/"pilot/summary",output/"pilot/figures")
    p1check=preservation(args.p1_baseline) if args.p1_baseline else {"status":"NOT_APPLICABLE", "reason":"No preexisting P1 inventory supplied"}
    write_json(output/"p1_preservation_after_smoke.json",p1check)
    passed=pilot["execution_status"]=="COMPLETE" and verified["implementation_verification"]=="PASS" and p1check["status"] in ("PASS","NOT_APPLICABLE")
    receipt={"status":"PASS" if passed else "FAIL","P2_unit_tests":units,"P1_regression":p1,
             "independent_mc":{k:mc[k] for k in ("status","solver_calls","training_datasets","test_input_datasets","wall_seconds","cpu_seconds")},
             "pilot":pilot,"pilot_verification_extra_calls":verified["verification_extra_calls"],
             "pilot_statistical_comparison":summary["statistical_comparison"],"P1_preservation":p1check["status"],
             "batch_size_fixed":config["batch_size"],"full_estimate_seconds_linear":pilot["wall_seconds"]*config["repetitions"]/50,
             "total_smoke_wall_seconds":time.perf_counter()-start,"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             "full_results_seen":False,"separate_from_full_counts":True}
    write_json(output/"smoke.json",receipt)
    print({k:receipt[k] for k in ("status","full_estimate_seconds_linear","peak_rss_mib","P1_preservation")})
    return 0 if passed else 1
