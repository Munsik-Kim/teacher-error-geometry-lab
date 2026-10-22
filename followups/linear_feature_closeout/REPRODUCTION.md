# Reproduce the frozen-feature appendix on CPU

The appendix ships small PCA64 train/evaluation arrays, fixed errors and labels, IDs, the reused gold coefficient, four fitted heads and their predictions. Verification needs no model, GPU, original research checkout or scenario generator. It reproduces the frozen-feature appendix; the [P1/P2 paper-record verifier](https://github.com/Munsik-Kim/teacher-error-geometry-lab/blob/v0.1.0/docs/REPRODUCTION.md) is a separate path.

## Install once

Run from this appendix directory or the root of an extracted standalone archive. Use Python 3.11 and the supplied lock, including NumPy 1.26.4 and Matplotlib 3.10.9:

```bash
python3.11 -m venv .venv
.venv/bin/python -I -m pip install -r requirements-lock.txt
.venv/bin/python -I -m pip check
.venv/bin/python -I -B scripts/check_package.py
```

No package installation or `PYTHONPATH` setting is needed: the script resolves its local source and input paths. The package checker verifies its scoped inventory, excluding local environment/cache directories. Installation above is for a new user environment; scientific closeout execution reused the existing CPU environment. The local environment and historical receipt paths are documented in provenance, not required by these commands.

From a full repository checkout, the root README shows the equivalent commands with the `followups/linear_feature_closeout/` prefix. In particular, `--run` is relative to the current working directory; default `--inputs` is resolved from the appendix's own location.

## Verify the stored result

```bash
.venv/bin/python -I -B scripts/closeout.py verify \
  --run results/closure_posthoc_01 \
  --output /tmp/teg-feature-verify-NEW
```

A fresh worker loads saved coefficients and recalculates train/evaluation predictions. Independent reduced-QR references solve the gold target once and the two error-only targets together: two QR factorizations/solve calls, three target-fit objects. They check the four actual target fits without refitting those primary heads. Further checks cover row metadata, signed risk decomposition, sign-pair linearity, high-error span transfer and residual-vector versus quadratic MSE.

The historical closeout verifier passed [53 checks](results/verification_01/verification.json) with `atol=1e-10, rtol=1e-8`, fixed before these fits. Its exact differences and environment are recorded in the receipt. Verification is a numerical implementation check, not registered H1 evidence.

## Rebuild tables and figures with zero fits

```bash
.venv/bin/python -I -B scripts/closeout.py summarize \
  --run results/closure_posthoc_01 \
  --output /tmp/teg-feature-summary-NEW
```

This reads only saved raw predictions, labels and errors. It writes `summary.json`, arm/transfer CSVs and PNG/SVG figures, with no access to the original F2R code or cache. The optional Gaussian-anchor plot uses copied historical facts; it is not another Gaussian experiment. Numbers can be compared with the saved summary; figure bytes can depend on renderer/font versions.

## Optional tests or replay of the four fits

```bash
.venv/bin/python -I -B scripts/closeout.py test --output /tmp/teg-feature-tests-NEW
.venv/bin/python -I -B scripts/closeout.py run-posthoc --output /tmp/teg-feature-replay-NEW
```

The nine tests use small synthetic numeric arrays, separate from official task data. The optional replay performs exactly four teacher-target FP64 `lstsq(rcond=None)` fits with an intercept and reuses the supplied gold coefficient. It writes a fresh protocol/source/input freeze beside the output. Repeating the same arrays adds no independent scientific replicate. `summarize` performs zero fits; `verify` adds only the reference computations described above.

Every output must be a new, absent external path. Each worker has a 900-second resident wall limit, a 100,000,000-byte artifact limit and child BLAS threads fixed to one. These commands preserve original F1/F2/F2D/F2R records and do not resume their main/final experiments.

## What is stored

`inputs/data.npz`, read with `allow_pickle=False`, contains 2,048 training and 2,048 evaluation rows. The z arrays have 64 columns; errors have fixed high/low columns. The first coefficient is the intercept. IDs and canonical signatures preserve row identity; labels retain `score/sqrt(110/3)` normalization. No model weight or full 576D feature cache is included.

`results/closure_posthoc_01/raw.npz` contains four coefficient columns and predictions, original gold predictions, labels/errors and row metadata. `fit_metadata.json` identifies arm order, target/design hashes, rank and solver convention. The ratio `G=Delta/mean(e_eval²)` uses actual error energy, without clipping, epsilon or nominal `.25` substitution. Source, config, input and result manifests identify distinct objects. See the [report](FINAL_REPORT.md), [claim map](CLAIM_MAP.md) and [history](PROVENANCE_AND_HISTORY.md) for interpretation and exact accounting.
