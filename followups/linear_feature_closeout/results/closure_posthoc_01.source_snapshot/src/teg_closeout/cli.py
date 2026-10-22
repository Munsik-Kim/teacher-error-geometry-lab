"""A relative-path CPU CLI; no original project/model imports or data generator."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import unittest
import numpy as np
from .analysis import analyze_frozen,load_frozen,summarize_saved,verify_saved
from .supervision import supervise
ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csv_write(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def freeze(output,inputs):
    target=Path(output).parent/(Path(output).name+'.source_freeze.json')
    files=[]
    paths=[ROOT/'CLOSURE_PROTOCOL.json',ROOT/'scripts/closeout.py',*ROOT.glob('src/**/*.py'),*ROOT.glob('tests/*.py')]
    snapshot=Path(output).parent/(Path(output).name+'.source_snapshot')
    if snapshot.exists() or target.exists():raise FileExistsError('SOURCE_FREEZE_EXISTS')
    for p in sorted(paths):
        rel=p.relative_to(ROOT);dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        files.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
    receipt=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),files=files,
                 source_bundle_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                 protocol_sha256=sha(ROOT/'CLOSURE_PROTOCOL.json'),frozen_input_sha256=sha(inputs),
                 kind='posthoc source/input freeze before these four fits; original qualification already observed')
    dump(target,receipt);return target

def plots(summary,out):
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(exist_ok=True)
    # Figure1: actual fitted risk and teacher agreement, plus the ratio of means.
    arms=summary['arms'];labels=['gold','high+','high−','low+','low−']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(5);axes[0].bar(x-.18,[summary['gold']['truth_risk']]+[r['truth_risk'] for r in arms],width=.36,label='Truth MSE')
    axes[0].bar(x[1:]+.18,[r['teacher_agreement_mse'] for r in arms],width=.36,label='Teacher agreement MSE')
    axes[0].set_xticks(x,labels);axes[0].set_ylabel('MSE on reused qualification (2048 rows)');axes[0].legend()
    ratios=[r['transfer_ratio_G'] if r['transfer_ratio_G'] is not None else np.nan for r in summary['transfer']]
    axes[1].bar(['high','low'],ratios);axes[1].axhline(1,color='black',linestyle='--');axes[1].set_ylabel('G = sign-average risk increase / actual error MSE')
    for i,row in enumerate(summary['transfer']):axes[1].text(i,ratios[i],f"  E={row['eval_error_mse']:.6f}\n  G={ratios[i]:.6f}",ha='center',va='bottom',fontsize=9)
    axes[1].set_ylim(top=max([1.1]+[r*1.2 for r in ratios if np.isfinite(r)]))
    fig.suptitle('Post-hoc frozen-feature OLS mechanism audit — one train dataset');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/('posthoc_transfer.'+ext),dpi=160)
    plt.close(fig)
    # Figure2: signed cross terms are retained for every fit.
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
    ax.bar(x-.2,[r['prediction_change_energy'] for r in arms],width=.4,label='mean(d²)')
    ax.bar(x+.2,[r['signed_cross_term'] for r in arms],width=.4,label='2 mean((f_gold−y*) d)')
    ax.plot(x,[r['risk_difference'] for r in arms],'ko',label='Actual truth-risk difference')
    ax.set_xticks(x,labels[1:]);ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('MSE contribution on reused qualification');ax.legend()
    ax.set_title('Post-hoc identity: ΔR = displacement energy + signed cross')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/('signed_decomposition.'+ext),dpi=160)
    plt.close(fig)
    # Figure3: historical paper anchor, copied full-precision facts (not new fits).
    bp=ROOT/'audit/baseline_facts.json'
    if bp.exists():
        b=json.loads(bp.read_text());p2=b['P2'];rows=p2.get('methods',p2.get('results',[]))
        if isinstance(rows,dict):rows=[dict(method=k,**v) for k,v in rows.items()]
        if rows:
            fig,ax=plt.subplots(figsize=(7,4));names=[r.get('method',r.get('name')) for r in rows]
            theory=[float(r.get('theory',r.get('theory_value'))) for r in rows]
            measured=[float(r.get('measured_mean',r.get('mean'))) for r in rows]
            err=[1.96*float(r.get('MC_SE',r.get('mc_se',0))) for r in rows]
            x=np.arange(len(rows));ax.plot(x,theory,'kx',ms=9,label='Analytical expectation');ax.errorbar(x,measured,yerr=err,fmt='o',capsize=4,label='Recorded MC mean ±1.96 MC-SE')
            ax.set_xticks(x,names);ax.set_ylabel('Truth population MSE');ax.set_title('P2 historical Gaussian anchor: n27, λ=.45');ax.legend();fig.tight_layout()
            for ext in ['png','svg']:fig.savefig(out/('gaussian_anchor.'+ext),dpi=160)
            plt.close(fig)

def worker(action,inputs,run,output):
    started=time.monotonic();cpu=time.process_time();output=Path(output)
    if action=='run-posthoc':
        summary=analyze_frozen(inputs,output)
        with np.load(output/'raw.npz',allow_pickle=False) as f:raw={k:f[k] for k in f.files}
        rows=[]
        for split in ['train','eval']:
            pred=raw[split+'_predictions'];ids=raw.get(split+'_scenario_ids',np.arange(len(pred)).astype(str))
            for i in range(len(pred)):
                rows.append(dict(split=split,row=i,scenario_id=str(ids[i]),gold_prediction=float(raw['gold_'+split+'_prediction'][i]),
                                 high_plus=float(pred[i,0]),high_minus=float(pred[i,1]),low_plus=float(pred[i,2]),low_minus=float(pred[i,3])))
        csv_write(output/'row_predictions.csv',rows)
    elif action=='verify':
        output.mkdir(parents=True,exist_ok=False);result=verify_saved(inputs,run);dump(output/'verification.json',result)
    elif action=='summarize':
        output.mkdir(parents=True,exist_ok=False);summary=summarize_saved(run);dump(output/'summary.json',summary)
        csv_write(output/'arm_metrics.csv',summary['arms']);csv_write(output/'transfer_metrics.csv',summary['transfer']);plots(summary,output/'figures')
        dump(output/'summary_receipt.json',dict(primary_fits=0,reference_fits=0,original_project_read=False,raw_only=True))
    elif action=='test':
        output.mkdir(parents=True,exist_ok=False);log=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        dump(output/'unit_tests.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',unique_tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),log=log.getvalue(),fixture_scope='independent numeric arrays only; no official split/generator/model'))
        if not result.wasSuccessful():raise AssertionError(log.getvalue())
    else:raise ValueError(action)
    dump(output/'CPU_receipt.json',dict(action=action,wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
         peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,Python=sys.version,executable=sys.executable,NumPy=np.__version__,OS=platform.platform(),
         BLAS_threads=1,no_user_site_flag=sys.flags.no_user_site,old_project_modules_imported=[x for x in sys.modules if x.startswith(('teg_f2','teg_followup','teacher_error_geometry'))],
         Torch_imported='torch' in sys.modules,GPU_forward_calls=0,new_case_generation=0))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ['run-posthoc','verify','summarize','test']:
        p=sub.add_parser(name);p.add_argument('--output',required=True);p.add_argument('--inputs',default=str(ROOT/'inputs/data.npz'))
        if name in ['verify','summarize']:p.add_argument('--run',required=True)
    p=sub.add_parser('_worker');p.add_argument('--action',dest='worker_action',required=True);p.add_argument('--inputs',required=True);p.add_argument('--run');p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='_worker':return worker(args.worker_action,args.inputs,args.run,args.output)
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError(output)
    if args.command=='run-posthoc':freeze(output,args.inputs)
    command=[sys.executable,'-I','-B',str(ROOT/'scripts/closeout.py'),'_worker','--action',args.command,'--inputs',str(Path(args.inputs).resolve()),'--output',str(output)]
    if getattr(args,'run',None):command+=['--run',str(Path(args.run).resolve())]
    result=supervise(command,output.parent/(output.name+'.supervision.json'),wall_limit=900,artifact_root=ROOT,artifact_limit=100_000_000)
    print(json.dumps(dict(status=result['status'],output=str(output),wall_seconds=result['wall_seconds'])))
    raise SystemExit(result['status']!='COMPLETE')
