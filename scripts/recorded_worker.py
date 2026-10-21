"""Execute an untouched historical verifier from a newly restored snapshot."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["p1", "p2"], required=True)
parser.add_argument("--records", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
records, output = Path(args.records).resolve(), Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
run_id = {"p1": "p1_full_20261021_02", "p2": "p2_paper_anchor_20261021T065112Z"}[args.phase]
run = records / "runs" / run_id
snapshot = run / "provenance/source"
baseline = json.loads((records / "runs/p2_paper_anchor_20261021T065112Z/provenance/p1_before.json").read_text())
old_root = Path(next(e["path"] for e in baseline["files"] if e["path"].endswith("/docs/P1_REPORT.md"))).parent.parent
forbidden = [old_root, old_root.parent / "handoffs", old_root.parent / "teacher-error-geometry-p2-rebuild-20261021T065112Z"]
blocked = []


def audit(event, event_args):
    if event == "socket.connect":
        raise RuntimeError("Recorded verification is offline")
    if event == "open" and event_args and isinstance(event_args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(event_args[0])).absolute()
        if any(p == root or root in p.parents for root in forbidden):
            blocked.append(str(p))
            raise RuntimeError("Original research paths are forbidden during public replay")


sys.addaudithook(audit)
sys.dont_write_bytecode = True
sys.path.insert(0, str(snapshot / "src"))
import teacher_error_geometry
assert snapshot in Path(teacher_error_geometry.__file__).resolve().parents
start = time.perf_counter()
metadata_receipt = None
if args.phase == "p1":
    from teacher_error_geometry import verification as module
    from teacher_error_geometry.summary import summarize
    from teacher_error_geometry.plotting import figures
    helper_spec = importlib.util.spec_from_file_location("_teg_frozen_metadata", Path(__file__).with_name("frozen_cell_metadata.py"))
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    original_validate = module.validate_config
    recorded_cells = json.loads((run / "provenance/cells.json").read_text())

    def recorded_design(config):
        global metadata_receipt
        cells, groups = original_validate(config)
        cells, groups, metadata_receipt = helper.validated_frozen_plan(cells, groups, recorded_cells, config["tolerance"])
        return cells, groups

    # Keep exact raw equality and every scientific check in the frozen verifier.
    module.validate_config = recorded_design
    verify = module.verify
else:
    from teacher_error_geometry.p2 import verification as module
    from teacher_error_geometry.p2.analysis import summarize
    from teacher_error_geometry.p2.plotting import figures

    def relocated_preservation(path):
        metadata = json.loads(Path(path).read_text())
        changed = []
        import hashlib
        for entry in metadata["files"]:
            old = Path(entry["path"])
            if old_root in old.parents:
                restored = records / old.relative_to(old_root)
            elif old.parent == old_root.parent / "handoffs":
                restored = records / "handoffs" / old.name
            else:
                raise ValueError("Unmapped historical preservation path")
            if not restored.is_file() or restored.stat().st_size != entry["bytes"] or hashlib.sha256(restored.read_bytes()).hexdigest() != entry["sha256"]:
                changed.append(entry["path"])
        return {"status": "PASS" if not changed else "FAIL", "protected_file_count": len(metadata["files"]),
                "changed_or_missing": changed, "verification_mode": "relocated original paths; original bytes and criteria unchanged",
                "original_research_access": False}

    # Adapt only the old machine-path inventory lookup, not its numerical verifier.
    module.preservation = relocated_preservation
    verify = module.verify
verification = verify(run, output / "verification.json")
summary = summarize(run, output / "summary")
plots = figures(output / "summary", output / "figures")
receipt = {"phase": args.phase, "implementation_verification": verification["implementation_verification"],
           "statistical_comparison": summary["statistical_comparison"], "raw_rows": summary["raw_rows"],
           "wall_seconds": time.perf_counter() - start, "historical_code_root": str(snapshot),
           "original_access_denied": True, "denied_original_open_attempts": len(blocked),
           "full_sweep_executed": False, "figure_count": plots["figure_count"]}
receipt["recorded_cell_metadata"] = metadata_receipt
(output / "worker.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
raise SystemExit(0 if verification["implementation_verification"] == "PASS" else 1)
