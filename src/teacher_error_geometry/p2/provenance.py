"""P2 freeze and P1 preservation. Never write into a protected P1 run."""
import hashlib
import json
from pathlib import Path
import subprocess
from ..artifacts import ROOT, environment, read_json, sha256, utc_now, write_json


def preservation(baseline_path):
    b=read_json(baseline_path); changed=[]
    for entry in b["files"]:
        p=Path(entry["path"])
        if not p.exists() or p.stat().st_size!=entry["bytes"] or sha256(p)!=entry["sha256"]:
            changed.append(str(p))
    protected={e["path"] for e in b["files"]}
    current=set()
    for p in (ROOT/"runs").iterdir():
        if p.is_dir() and (p.name.startswith("p0_") or p.name.startswith("p1_")):
            current.update(str(f) for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    current.update(str(f) for f in (ROOT/"reproductions").rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    added=sorted(current-protected)
    return {"status":"PASS" if not changed and not added else "FAIL","protected_file_count":len(protected),
            "protected_bytes":sum(e["bytes"] for e in b["files"]),"changed_or_missing":changed,
            "added_in_protected_run_or_rebuild":added,"baseline_git_head":b["git_head"],
            "baseline_manifest_sha256":sha256(baseline_path),"checked_at_utc":utc_now(),
            "scope":"All preexisting P0/P1 run files, snapshots, receipts, rebuilds, P1 reports/theory/source_access/history/config/tests/science modules and old handoff; shared CLI may be extended"}


def freeze(run,config,started_at,baseline_path=None):
    p=run/"provenance"; p.mkdir()
    write_json(p/"config.json",config); write_json(p/"environment.json",environment())
    paths=[]
    for directory in ("src","tests","tests_p2","scripts","configs"):
        paths.extend(f for f in (ROOT/directory).rglob("*") if f.is_file() and "__pycache__" not in f.parts and f.suffix!=".pyc")
    for name in ("pyproject.toml","requirements.txt","environment.json","LICENSE","NOTICE",".gitattributes",
                 "docs/protocol.md","docs/claim_map.md","docs/P2_PROTOCOL.md","docs/P2_SOURCE_MAP.ko.md"):
        paths.append(ROOT/name)
    manifest=[]
    for source in sorted(set(paths)):
        relative=source.relative_to(ROOT).as_posix(); data=source.read_bytes()
        target=p/"source"/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        manifest.append({"path":relative,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True)
    status=subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True)
    result={"stage":"P2","source_bundle_sha256":digest,"config_sha256":sha256(p/"config.json"),
            "started_at_utc":started_at,"source_manifest":manifest,
            "source_manifest_definition":"SHA256 canonical JSON manifest (sort_keys=True,separators=(',',':')), entries sorted by path",
            "git_commit":git.stdout.strip() if git.returncode==0 else None,
            "git_status_at_start":status.stdout if status.returncode==0 else "UNVERSIONED_COPY",
            "author_code_access":"NONE","reports_written_after_execution_are_not_frozen_science":True}
    if baseline_path:
        check=preservation(baseline_path)
        if check["status"]!="PASS":
            raise RuntimeError("P1 protection check failed")
        (p/"p1_before.json").write_bytes(Path(baseline_path).read_bytes())
        write_json(p/"p1_preservation_at_start.json",check)
        result["p1_baseline_sha256"]=sha256(p/"p1_before.json")
    write_json(p/"freeze.json",result)
    return result
