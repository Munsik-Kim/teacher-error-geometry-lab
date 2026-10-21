import argparse
import io
from pathlib import Path
import resource
import time
import unittest
from .artifacts import ROOT, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Teacher Error Geometry Lab: CPU reproduction")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--config", default=str(ROOT / "configs/p1.json"))
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", required=True)
    verify.add_argument("--output")
    summary = sub.add_parser("summarize")
    summary.add_argument("--run", required=True)
    summary.add_argument("--output", required=True)
    figure = sub.add_parser("figures")
    figure.add_argument("--summary", required=True)
    figure.add_argument("--output", required=True)
    from .p2.cli import register as register_p2
    register_p2(sub)
    for name in ("verify-public", "restore-recorded"):
        public = sub.add_parser(name)
        public.add_argument("--output", required=True, help="New external output directory")
    args = parser.parse_args()
    if args.command == "restore-recorded":
        from .public_archive import restore
        print(restore(ROOT, args.output))
        return 0
    if args.command == "verify-public":
        from .public_validation import verify_public
        result = verify_public(ROOT, args.output)
        print({k: result[k] for k in ("public_reproduction", "raw_rows_checked", "wall_seconds")})
        return 0
    if args.command == "p2":
        from .p2.cli import dispatch as dispatch_p2
        return dispatch_p2(args)
    if args.command == "run":
        from .experiment import execute
        result = execute(read_json(args.config), args.output)
        print({k:result[k] for k in ("execution_status", "wall_seconds", "actual")})
        return 0 if result["execution_status"] == "COMPLETE" else 2
    if args.command == "verify":
        from .verification import verify as perform_verify
        result = perform_verify(args.run, args.output)
        print(result)
        return 0 if result["implementation_verification"] == "PASS" else 1
    if args.command == "summarize":
        from .summary import summarize
        result = summarize(args.run, args.output)
        print({k: result[k] for k in ("cells", "raw_rows", "failed_rows", "statistical_comparison")})
        return 0
    if args.command == "figures":
        from .plotting import figures
        result = figures(args.summary, args.output)
        print({"figure_count": result["figure_count"], "files": len(result["files"])})
        return 0
    if args.command == "smoke":
        from .checks import independent_mc_checks
        from .design import small_config
        from .experiment import execute
        from .summary import summarize
        from .verification import verify as perform_verify
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        stream = io.StringIO()
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (output / "unittest.txt").write_text(stream.getvalue(), encoding="utf-8")
        print(stream.getvalue(), flush=True)
        if not tests.wasSuccessful():
            write_json(output / "smoke.json", {"status": "FAIL", "tests_run": tests.testsRun})
            return 1
        mc = independent_mc_checks()
        write_json(output / "independent_test_mc.json", mc)
        if not mc["passed"]:
            write_json(output / "smoke.json", {"status": "FAIL", "reason": "independent test MC"})
            return 1
        config = read_json(args.config)
        pilot = execute(small_config(config), output / "pilot", progress=False)
        verified = perform_verify(output / "pilot")
        summarized = summarize(output / "pilot", output / "pilot/summary")
        passed = verified["implementation_verification"] == "PASS" and pilot["execution_status"] == "COMPLETE"
        result = {"status": "PASS" if passed else "FAIL", "tests_run": tests.testsRun,
                  "independent_test_mc_passed": mc["passed"], "pilot_execution": pilot,
                  "pilot_implementation_verification": verified["implementation_verification"],
                  "pilot_statistical_comparison": summarized["statistical_comparison"],
                  "full_estimated_wall_seconds_linear": pilot["wall_seconds"] * config["repetitions"] / 20,
                  "pilot_batch_size": config["batch_size"], "pilot_repetitions": 20,
                  "batch_size_decision": "200, bounded serial writing; no arrays retained across datasets",
                  "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  "total_smoke_wall_seconds": time.perf_counter() - started,
                  "full_results_seen": False}
        write_json(output / "smoke.json", result)
        print({k: result[k] for k in ("status", "tests_run", "full_estimated_wall_seconds_linear", "peak_rss_mib")})
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
