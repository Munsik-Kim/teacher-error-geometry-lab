# Teacher Error Geometry Lab

**How does the direction of a teacher's error affect a student's error against the true target?** This project independently implements existing linear teacher–student regression theory and compares analytical risks with actual fitted students. It provides Gaussian generators, teacher targets, FP64 OLS, population-risk calculations, paired summaries and tools that rebuild results from saved records. **248,000 canonical fitted-student records** are preserved for CPU verification.

**Munsik Kim** · [한국어](README.ko.md) · [CPU workflow](https://github.com/Munsik-Kim/teacher-error-geometry-lab/actions/workflows/cpu.yml) · [Releases](https://github.com/Munsik-Kim/teacher-error-geometry-lab/releases)

![P1 truth risk by sample size: analytical curves and recorded means with pointwise 95% intervals](recorded/p1/figures/01_truth_by_n.png)

In P1 at rho=0 and n=30, visible teacher error gives student truth MSE **0.250000** even when teacher-agreement MSE is effectively zero. Hidden error gives **0.008461**, against an analytical expectation of **0.008929**. Agreement with the teacher and accuracy against truth are different measurements.

## What was reproduced

The source is Agus Sudjianto and Denis Burakov, [*Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models*, arXiv:2610.26797v1](https://arxiv.org/html/2610.26797v1).

- **P1:** a controlled two-dimensional Gaussian special case of §4.2. The 84 settings contain 168,000 canonical fits on 62,000 datasets.
- **P2:** the d=5, r=3, n=27, lambda=0.45 Gaussian setting from Table 1. Four regression methods share 20,000 datasets and produce 80,000 fits.

| P2 method | Analytical expected truth risk | Recorded mean |
|---|---:|---:|
| Full input / outcome (FO) | 0.309524 | 0.310155 |
| Full input / teacher (FT) | 0.404467 | 0.403867 |
| Reduced input / outcome (RO) | 0.260000 | 0.259864 |
| Reduced input / mixed target (RM) | 0.206875 | 0.207679 |

The implementation separates fitted risk from expected risk and checks them through independent calculations. Historical verification covers **36 CPU tests and 248,000 canonical fit rows**. [Results](docs/RESULTS.md), [full-precision facts](data/results.json) and the [paper-to-code map](docs/claim_map.md) preserve the exact values and assumptions.

## Frozen Transformer feature appendix

The [linear-feature appendix](followups/linear_feature_closeout/README.md) uses saved SmolLM2-135M features to examine the same linear readout mechanism on nonGaussian inputs. Its diagnostics found that expanding PCA16 to PCA64 improved gold-only validation MSE from **0.300690 to 0.175955**. A later independent qualification stopped because a teacher-error norm missed its registered control range.

A separate four-fit post-hoc analysis retained the original errors.

| Error direction | Truth risk, ± average | Error-energy transfer ratio G |
|---|---:|---:|
| High | 0.4164 | 1.0000 |
| Low | 0.1992 | 0.0619 |

These are descriptive results for one training dataset and a reused qualification set; G is an energy ratio, not accuracy. The registered H1 main experiment and final evaluation remain unrun, and neither earlier stop becomes a pass.

## Two CPU reproduction paths

Run from the repository root with Python 3.11. Install the declared CPU environment once; neither path needs model weights, a GPU or the original research checkout.

```bash
python3.11 -m venv .venv
.venv/bin/python -I -m pip install -r requirements-lock.txt
.venv/bin/python -I -m pip install --no-build-isolation --no-deps .
.venv/bin/python -I -m pip check
```

**Paper records:** restore the evidence, check saved P1/P2 rows, replay selected seeds and rebuild summaries/figures without rerunning the full sweeps.

```bash
.venv/bin/python -I -B -m teacher_error_geometry verify-public --output /tmp/teg-public-NEW
```

**Feature appendix:** verify frozen coefficients and predictions with independent QR calculations, then rebuild its tables and figures with zero new fits.

```bash
.venv/bin/python -I -B followups/linear_feature_closeout/scripts/check_package.py
.venv/bin/python -I -B followups/linear_feature_closeout/scripts/closeout.py verify --run followups/linear_feature_closeout/results/closure_posthoc_01 --output /tmp/teg-feature-verify-NEW
.venv/bin/python -I -B followups/linear_feature_closeout/scripts/closeout.py summarize --run followups/linear_feature_closeout/results/closure_posthoc_01 --output /tmp/teg-feature-summary-NEW
```

Use a new, absent external output path for every command. The appendix has nine synthetic-array tests and 53 recorded numerical checks; these are separate from the historical 36-test paper verifier.

## Read the evidence

- [Paper reproduction guide](docs/REPRODUCTION.md) · [Theory](docs/theory.md) · [Original execution history](docs/HISTORY.md)
- [Final linear-reproduction report](followups/linear_feature_closeout/FINAL_REPORT.md) · [Appendix reproduction guide](followups/linear_feature_closeout/REPRODUCTION.md)
- [Consolidated provenance and access history](followups/linear_feature_closeout/PROVENANCE_AND_HISTORY.md) · [Publication history](docs/PUBLICATION_HISTORY.md)

The scope is Gaussian regression and linear students on fixed pretrained features; detailed assumptions and unrun extensions are collected in the [consolidated history](followups/linear_feature_closeout/PROVENANCE_AND_HISTORY.md). NumPy/lstsq and Matplotlib are upstream, and implementation/internal checks were AI-assisted. [MIT](LICENSE) covers independent project work; paper and upstream rights remain separate ([NOTICE](NOTICE)).
