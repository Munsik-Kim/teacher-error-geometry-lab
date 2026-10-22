# Linear feature appendix

**How does a fixed linear student inherit a teacher's error?** This appendix completes the linear reproduction portfolio with real pretrained features: it reuses frozen SmolLM2-135M PCA64 arrays, fits four corrupted-oracle targets with FP64 OLS, and independently checks the resulting truth risks, teacher agreement and error transfer.

[한국어](README.ko.md) · [Project overview](https://github.com/Munsik-Kim/teacher-error-geometry-lab) · [Final report](FINAL_REPORT.md)

![Truth risk, teacher agreement and error-energy transfer on reused qualification data](results/summary_01/figures/posthoc_transfer.png)

| Error direction | Truth risk R, ± average | Agreement A, ± average | Actual error E | Risk increase Delta | G=Delta/E |
|---|---:|---:|---:|---:|---:|
| High | 0.416396 | 0.183002 | 0.233394 | 0.233394 | 1.0000 |
| Low | 0.199230 | 0.445555 | 0.262229 | 0.016228 | 0.0619 |

*Post-hoc measurements from one training dataset and its already-accessed 2,048-row qualification set; the original errors were not rescaled. Gold truth risk is 0.183002. Values are rounded here; [facts.json](facts.json) retains full precision and individual ± arms.*

G is a ratio of mean energies. High error belongs to the student's fixed linear span, so G≈1 checks a known OLS property. The low-error result was measured. These are not probabilities, population projection fractions or a registered H1 conclusion.

## Connection to the paper reproduction

P1 independently implements the two-dimensional Gaussian special case of §4.2; P2 reproduces the d5/r3/n27/lambda.45 Gaussian setting from Table 1 of [Sudjianto and Burakov, arXiv:2610.26797v1](https://arxiv.org/html/2610.26797v1). [The project overview](https://github.com/Munsik-Kim/teacher-error-geometry-lab) links their 248,000 canonical rows and historical 36-test verification. This appendix adds fixed-feature linear mechanism checks on nonGaussian inputs, rather than applying Gaussian finite-sample formulas to Transformer risks.

The earlier PCA16 task validation stopped; cache-only diagnostics supported using PCA64. A later independent qualification stopped at the error-norm control. Those outcomes remain intact. This four-fit analysis is separate and post hoc; the registered H1 main/final, H2 and H3 remain **NOT_RUN**.

## CPU verification and summaries

Run from this directory or an extracted standalone package root with Python 3.11. Install the CPU environment once:

```bash
python3.11 -m venv .venv
.venv/bin/python -I -m pip install -r requirements-lock.txt
.venv/bin/python -I -m pip check
.venv/bin/python -I -B scripts/check_package.py
.venv/bin/python -I -B scripts/closeout.py verify --run results/closure_posthoc_01 --output /tmp/teg-feature-verify-NEW
.venv/bin/python -I -B scripts/closeout.py summarize --run results/closure_posthoc_01 --output /tmp/teg-feature-summary-NEW
```

`verify` reloads coefficients and checks predictions through independent QR references; recorded verification passed 53 checks. `summarize` performs zero fits. Use a new external output path. No model, GPU, original research path or official-data generator is needed. [REPRODUCTION.md](REPRODUCTION.md) describes the nine synthetic-array tests and optional four-fit replay. [Original P1/P2 stored-record reproduction](https://github.com/Munsik-Kim/teacher-error-geometry-lab/blob/v0.1.0/docs/REPRODUCTION.md) uses a separate path.

[Claim map](CLAIM_MAP.md) · [History and access boundaries](PROVENANCE_AND_HISTORY.md) · [Full-precision facts](facts.json). The scope is a frozen linear readout and synthetic teacher; nonlinear students, learned representations, natural LLM mistakes and generative distillation are not tested. Implementation and checks were AI-assisted; historical failures and exposure records remain traceable in the history.
