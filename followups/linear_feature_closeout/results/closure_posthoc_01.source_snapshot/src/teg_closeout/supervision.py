"""Hard resident-wall supervision of newly created, owned process groups only."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
from datetime import datetime, timezone

def now():
    return datetime.now(timezone.utc).isoformat()

def logical_bytes(root):
    root=Path(root)
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()) if root.exists() else 0

def supervise(command, receipt, wall_limit=900, ready_path=None,
              ready_delay_limit=None, artifact_root=None, artifact_limit=5_000_000_000):
    receipt=Path(receipt)
    stdout=receipt.with_suffix('.stdout.log'); stderr=receipt.with_suffix('.stderr.log')
    if any(p.exists() for p in [receipt,stdout,stderr]):raise FileExistsError('SUPERVISOR_OUTPUT_EXISTS')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
        env[key]='1'
    env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    start=time.monotonic();started=now();readied=None;ready_utc=None;reason=None;term=False;kill=False;peak=0
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with stdout.open('xb') as out,stderr.open('xb') as err:
        proc=subprocess.Popen(command,stdout=out,stderr=err,env=env,start_new_session=True,shell=False)
        while proc.poll() is None:
            elapsed=time.monotonic()-start
            if ready_path and readied is None and Path(ready_path).exists():readied=time.monotonic();ready_utc=now()
            if artifact_root:
                peak=max(peak,logical_bytes(artifact_root))
                if peak>artifact_limit:reason='ARTIFACT_LIMIT'
            if elapsed>=wall_limit:reason='WALL_TIMEOUT'
            if readied is not None and ready_delay_limit is not None and time.monotonic()-readied>=ready_delay_limit:reason='READY_TIMEOUT_INJECTION'
            if reason:
                try:os.killpg(proc.pid,signal.SIGTERM);term=True
                except ProcessLookupError:pass
                try:proc.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL);kill=True
                    except ProcessLookupError:pass
                    proc.wait()
                break
            time.sleep(.02)
        code=proc.wait()
    wall=time.monotonic()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
    if artifact_root:peak=max(peak,logical_bytes(artifact_root))
    result=dict(status='COMPLETE' if code==0 and reason is None else 'BUDGET_BLOCKED' if reason else 'WORKER_FAILED',
                command=command,started_utc=started,finished_utc=now(),wall_seconds=wall,
                wall_limit_seconds=wall_limit,ready_seen=readied is not None,ready_seen_utc=ready_utc,
                ready_to_exit_seconds=None if readied is None else time.monotonic()-readied,
                reason=reason,exit_code=code,owned_pid=proc.pid,owned_process_group=proc.pid,
                TERM_sent=term,KILL_sent=kill,partial_outputs_preserved=True,
                worker_CPU_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
                cumulative_child_peak_RSS_KiB=after.ru_maxrss,
                RSS_scope='process children cumulative max, not summed tree or isolated per-worker peak; worker own receipt when available',
                sampled_artifact_bytes=peak,artifact_limit_bytes=artifact_limit,
                stdout=str(stdout),stderr=str(stderr),other_processes_signalled=False)
    with receipt.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    return result
