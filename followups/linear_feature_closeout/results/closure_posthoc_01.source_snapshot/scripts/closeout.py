#!/usr/bin/env python3
"""Four-fit post-hoc execution, independent verification, fit-free summaries."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg';os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplcache'))
sys.path.insert(0,str(ROOT/'src'))
from teg_closeout.cli import main
if __name__=='__main__':main()
