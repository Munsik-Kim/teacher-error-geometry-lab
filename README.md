# Teacher Error Geometry Lab

A CPU reproduction of teacher–student regression theory

[한국어](README.ko.md)

An independent CPU implementation of linear teacher–student regression theory. Controlled experiments vary input correlation and the information available to the student. A separate experiment reproduces one Gaussian regression setting from the paper and compares measured population risks with analytical expectations.

We examine how teacher errors of the same population size reach the student's true-signal risk. The implementation covers Gaussian data and teacher targets, FP64 unregularized OLS, population risk, paired statistical comparisons, and recorded-data reproduction tools. NumPy/lstsq and Matplotlib are upstream components.

![P1 truth risk by sample size: theory curves and measured means with pointwise 95% intervals](recorded/p1/figures/01_truth_by_n.png)

At rho=0 and n=30, visible teacher error gives measured truth risk 0.25 despite teacher-agreement MSE about 1.5e-31. Hidden error gives 0.00846083145254395 (expectation 0.008928571428571428). P1 contains 84 settings × 2,000 repetitions, 168,000 fits on 62,000 datasets. P2 uses 20,000 shared datasets for 80,000 fits; RM has the smallest measured mean in this fixed setting, 0.20767895944476805 (expectation 0.20687515217391308). See [results and limits](docs/RESULTS.md) and [full-precision facts](data/results.json).

P1 is an independently controlled two-dimensional special case of §4.2; P2 partially reproduces the n=27, lambda=.45 Gaussian setting in Table 1 of [Sudjianto and Burakov, arXiv:2610.26797v1](https://arxiv.org/html/2610.26797v1). See the [claim map](docs/claim_map.md) and [history](docs/HISTORY.md). The scope excludes the full paper, learned representations, LLMs, probit and CFPB. Implementation and internal checks were AI-assisted; no external human review or bitwise copy of the paper's PyTorch samples is claimed.

## Install and verify stored results

Run from the repository root with Python 3.11. Installation downloads only the declared CPU dependencies; verification is offline and requires no original research directory. Use a fresh external output name each time.

```bash
python3.11 -m venv .venv
.venv/bin/python -I -m pip install -r requirements-lock.txt
.venv/bin/python -I -m pip install --no-build-isolation --no-deps .
.venv/bin/python -I -m pip check
.venv/bin/python -I -B -m teacher_error_geometry verify-public --output /tmp/teg-public-NEW_ID
```

The single command runs 17 P1, 12 P2 and 6 archive/tool tests, restores the lossless evidence, checks all 248,000 canonical fit rows, replays selected seeds, and regenerates summaries plus seven PNG/SVG figures. Outputs and new receipts stay outside the repository. [Reproduction guide](docs/REPRODUCTION.md) explains restore-only and individual summary/figure commands. [P3 validation](docs/P3_REPORT.md) distinguishes same-machine clean-copy checks from external reproduction. CI is prepared locally; remote CI and publication have not run.

## Optional new experiments

These commands are separate from stored-result verification. They create new runs and are not part of P3.

```bash
scripts/teg smoke --output /tmp/teg-p1-smoke-NEW_ID
scripts/teg p2 smoke --output /tmp/teg-p2-smoke-NEW_ID
scripts/teg run --config configs/p1.json --output /tmp/teg-p1-full-NEW_ID
scripts/teg p2 run --config configs/p2_paper_anchor.json --output /tmp/teg-p2-full-NEW_ID
```

`scripts/teg` selects explicit `TEG_PYTHON`, then project `.venv`, then active `python3`; install the package into that interpreter first. It limits child BLAS threads to one. [MIT license](LICENSE) covers independent project work; the paper retains its own rights ([NOTICE](NOTICE)). Local public candidate only; no remote repository or release is asserted.
