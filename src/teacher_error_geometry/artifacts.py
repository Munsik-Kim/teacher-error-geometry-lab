import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import matplotlib

from .paths import project_root
ROOT = project_root()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fields = fieldnames or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def environment():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines() if Path("/proc/cpuinfo").exists() else []
    cpu = next((s.split(":", 1)[1].strip() for s in cpu_lines if s.startswith("model name")), platform.processor())
    names = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "MPLCONFIGDIR", "MPLBACKEND"]
    return {"python_executable": sys.executable, "python": sys.version,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "os": platform.platform(), "cpu": cpu, "visible_cpu_count": os.cpu_count(),
            "blas_build_configuration": buffer.getvalue(),
            "child_environment": {name: os.environ.get(name) for name in names},
            "dtype": "float64", "captured_at": utc_now()}


def freeze(run, config, started_at):
    provenance = run / "provenance"
    provenance.mkdir()
    write_json(provenance / "config.json", config)
    write_json(provenance / "environment.json", environment())
    paths = []
    for directory in ("src", "tests", "scripts", "configs"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and ".pyc" != p.suffix)
    paths.extend(ROOT / p for p in ("pyproject.toml", "requirements.txt", "environment.json",
                 "docs/theory.md", "docs/protocol.md", "docs/claim_map.md", "docs/source_access.md"))
    manifest = []
    for path in sorted(set(paths)):
        if not path.exists():
            raise FileNotFoundError(f"Required freeze input missing: {path}")
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        target = provenance / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    source_hash = hashlib.sha256(canonical).hexdigest()
    cfg_hash = sha256(provenance / "config.json")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    result = {"source_bundle_sha256": source_hash, "config_sha256": cfg_hash,
              "source_manifest": manifest, "started_at_utc": started_at,
              "source_manifest_definition": "SHA256 of canonical JSON list (sort_keys=True,separators=(',',':')); ordered by path",
              "git_commit": git.stdout.strip() if git.returncode == 0 else None,
              "git_status_at_start": status.stdout, "author_code_access": "NONE",
              "source_snapshot": "provenance/source/; reports written later are not execution source",
              "source_version_is": "bundle hash plus frozen config; git commit is supplementary"}
    write_json(provenance / "freeze.json", result)
    return result
