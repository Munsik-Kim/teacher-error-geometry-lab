"""One offline recorded-data check; no P1/P2 full experiment is launched."""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import site
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
from .public_archive import check_tree, digest, load_manifest, restore

TOL = {"atol": 1e-10, "rtol": 1e-9}
EXPECTED = {
    "p1": ("df1173e91e77a43a53e9a82beea079c65bd0dab6", "286b7639b3abfb69503926dad32f35dc27ddd11356a8a15fe2b72377e37716a2", "f909d298b234af6a3a1a4175066004bd203d91d0ac9912cb34271e058551a11b"),
    "p2": ("9b502ee4739e7acb43f94a89bdf57cfbc303a38b", "fc3f8e3e43beb117efd16c391d89a0d7b3d103611613d78639c869a01ddd20c7", "510781991629e037bd2dc1d66aa4533c7d903b3315b04050a53ee9cca2c1b4ef"),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def compare_values(left, right, stats, label="root"):
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("Summary keys differ: " + label)
        for key in left:
            compare_values(left[key], right[key], stats, label + "/" + key)
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError("Summary lengths differ: " + label)
        for index, (a, b) in enumerate(zip(left, right)):
            compare_values(a, b, stats, label + "/" + str(index))
    else:
        # CSV numeric fields are strings; metadata and blank deterministic z remain literal.
        numeric = not isinstance(left, bool) and left is not None
        try:
            a, b = float(left), float(right)
        except (TypeError, ValueError):
            numeric = False
        if numeric:
            if not math.isfinite(a) or not math.isfinite(b) or not np.isclose(a, b, **TOL):
                raise ValueError("Summary numeric mismatch: " + label)
            stats["numeric_values_compared"] += 1
            stats["max_absolute_difference"] = max(stats["max_absolute_difference"], abs(a-b))
        elif left != right or type(left) != type(right):
            raise ValueError("Summary value mismatch: " + label)


def compare_summaries(old, new):
    old, new = Path(old), Path(new)
    names = {p.name for p in old.iterdir() if p.suffix in (".json", ".csv")}
    if names != {p.name for p in new.iterdir() if p.suffix in (".json", ".csv")}:
        raise ValueError("Summary file set mismatch")
    result = {"numeric_values_compared": 0, "max_absolute_difference": 0., "files": []}
    for name in sorted(names):
        if name.endswith(".json"):
            a, b = read(old/name), read(new/name)
        else:
            with (old/name).open(newline="", encoding="utf-8") as x, (new/name).open(newline="", encoding="utf-8") as y:
                cx, cy = csv.DictReader(x), csv.DictReader(y)
                if cx.fieldnames != cy.fieldnames:
                    raise ValueError("CSV fields differ: " + name)
                a, b = list(cx), list(cy)
        compare_values(a, b, result, name)
        result["files"].append({"path": name, "bytes_identical": digest(old/name) == digest(new/name), "sha256": digest(new/name)})
    result.update(status="PASS", tolerance=TOL)
    return result


def frozen_sources(records, manifest):
    checked = {}
    for phase in ("p1_first", "p1", "p2"):
        run = records/"runs"/manifest["runs"][phase]
        frozen = read(run/"provenance/freeze.json")
        canonical = json.dumps(frozen["source_manifest"], sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest() != frozen["source_bundle_sha256"]:
            raise ValueError("Frozen source bundle digest mismatch: " + phase)
        for entry in frozen["source_manifest"]:
            path = run/"provenance/source"/entry["path"]
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError("Frozen source bytes differ: " + phase + "/" + entry["path"])
        if digest(run/"provenance/config.json") != frozen["config_sha256"]:
            raise ValueError("Frozen config differs: " + phase)
        if phase in EXPECTED and (frozen["git_commit"], frozen["source_bundle_sha256"], frozen["config_sha256"]) != EXPECTED[phase]:
            raise ValueError("Historical execution identities differ: " + phase)
        checked[phase] = {k: frozen[k] for k in ("git_commit", "source_bundle_sha256", "config_sha256")}
        checked[phase]["snapshot_files_checked"] = len(frozen["source_manifest"])
    first, final = (records/"runs"/manifest["runs"][p] for p in ("p1_first", "p1"))
    if read(first/"provenance/raw_manifest.json") != read(final/"provenance/raw_manifest.json"):
        raise ValueError("P1 same-sample rerun raw manifests differ")
    checked["p1_raw_identical_across_two_full_runs"] = True
    checked["past_commits_present_in_public_git_history"] = False
    return checked


def verify_public(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output == project or project in output.parents:
        raise ValueError("Use a new output path outside the public repository")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); cpu = time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "TEG_PROJECT_ROOT"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(output/"matplotlib-cache"), TMPDIR=str(output/"tmp"))
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    (output/"tmp").mkdir()
    manifest, entries = load_manifest(project)
    restored = restore(project, output/"records")
    records = output/"records"
    frozen = frozen_sources(records, manifest)
    copies = 0
    for entry in entries.values():
        if entry.get("public_copy"):
            path = project/entry["public_copy"]
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError("Normal-file historical copy differs: " + entry["public_copy"])
            copies += 1

    def child(command, log):
        tick = time.perf_counter()
        run = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True)
        (output/log).write_text(run.stdout+run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError("Public check failed; inspect " + str(output/log))
        return run, time.perf_counter()-tick

    suites = []
    for directory, expected in (("tests", 17), ("tests_p2", 12), ("tests_public", 6)):
        result, elapsed = child([sys.executable, "-I", "-B", "-m", "unittest", "discover", "-s", str(project/directory), "-v"], directory+".log")
        count = re.search(r"Ran (\d+) tests?", result.stderr)
        if count is None or int(count.group(1)) != expected:
            raise ValueError("Unexpected test denominator: " + directory)
        suites.append({"suite": directory, "tests_run": expected, "status": "PASS", "wall_seconds": elapsed})
    phases = {}
    for phase in ("p1", "p2"):
        child([sys.executable, "-I", "-B", str(project/"scripts/recorded_worker.py"), "--phase", phase,
               "--records", str(records), "--output", str(output/phase)], phase+".log")
        info = read(output/phase/"worker.json")
        if info["implementation_verification"] != "PASS" or info["denied_original_open_attempts"]:
            raise ValueError("Verification or original-path isolation failed: " + phase)
        run = records/"runs"/manifest["runs"][phase]
        info["summary_comparison"] = compare_summaries(run/"summary", output/phase/"summary")
        info["figures"] = []
        for image in sorted((output/phase/"figures").iterdir()):
            if image.suffix not in (".png", ".svg"):
                continue
            old = run/"figures"/image.name
            info["figures"].append({"path": image.name, "sha256": digest(image), "bytes_identical": digest(old) == digest(image)})
        phases[phase] = info
    # Compare against the common historical facts, not hand-rounded README numbers.
    for phase, fact_path in (("p1", "docs/report_facts.json"), ("p2", "docs/P2_report_facts.json")):
        old = read(records/fact_path)
        public = read(project/"data"/(phase+"_historical_facts.json"))
        compare_values(old, public, {"numeric_values_compared": 0, "max_absolute_difference": 0.}, phase+" facts")
    immutable_after = check_tree(records, entries)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    import teacher_error_geometry
    import matplotlib
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")}
    packages["project_import"] = str(Path(teacher_error_geometry.__file__).resolve())
    packages["numpy_import"] = str(Path(np.__file__).resolve())
    packages["matplotlib_import"] = str(Path(matplotlib.__file__).resolve())
    if site.ENABLE_USER_SITE or not sys.flags.isolated or sys.prefix == sys.base_prefix:
        raise ValueError("Public verification requires isolated Python in a project venv (-I)")
    for key in ("project_import", "numpy_import", "matplotlib_import"):
        if Path(sys.prefix) not in Path(packages[key]).parents:
            raise ValueError("Dependency imported outside the selected venv: " + key)
    receipt = {"scientific_scope": "P1_SPECIAL_CASE_AND_P2_SINGLE_GAUSSIAN_SETTING",
               "public_reproduction": "PASS", "recorded_integrity": "PASS", "frozen_source_checks": frozen,
               "restored": restored, "restored_unchanged_after_verification": immutable_after,
               "historical_normal_file_copies_checked": copies, "tests": suites, "phases": phases,
               "raw_rows_checked": sum(p["raw_rows"] for p in phases.values()),
               "tolerance": TOL, "python": sys.version, "packages": packages,
               "isolated_python": bool(sys.flags.isolated), "user_site_enabled": site.ENABLE_USER_SITE,
               "same_machine_and_OS_as_original": True, "external_independent_reproduction": False,
               "new_full_sweeps": 0, "remote_publication": "NOT_RUN", "network_required_for_recorded_check": False,
               "wall_seconds": time.perf_counter()-start,
               "cpu_seconds_including_children": time.process_time()-cpu + child_after.ru_utime+child_after.ru_stime-child_before.ru_utime-child_before.ru_stime,
               "peak_child_rss_mib": child_after.ru_maxrss/1024,
               "AI_assisted_internal_check": True, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write(output/"report.json", receipt)
    return receipt
