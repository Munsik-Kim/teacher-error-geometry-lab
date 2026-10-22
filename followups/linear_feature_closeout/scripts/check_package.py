#!/usr/bin/env python3
"""Read-only verification of a portable package's own scoped inventory."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.mplcache','__pycache__','.venv'}
def main():
    manifest=ROOT/'package_manifest.json'
    if not manifest.exists():manifest=ROOT/'manifest.json'
    expected=json.loads(manifest.read_text())['files']
    actual=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p==manifest or EXCLUDED.intersection(rel.parts):continue
        if p.is_symlink():raise ValueError('SYMLINK_NOT_ALLOWED:'+str(rel))
        if p.is_file():actual.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    actual.sort(key=lambda row:row['path'])
    good=actual==expected
    print(json.dumps(dict(status='PASS' if good else 'FAIL',file_count_excluding_self=len(actual),bytes_excluding_self=sum(r['bytes'] for r in actual))))
    raise SystemExit(not good)
if __name__=='__main__':main()
