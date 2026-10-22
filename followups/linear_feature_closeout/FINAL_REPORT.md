# Linear theory reproduction: final report

## The question and the implementation

How much of a teacher's error is inherited by a student with a fixed representation? This project turns existing linear teacher–student regression theory into executable, verifiable experiments. It independently implements Gaussian data generation, teacher targets, FP64 OLS, fitted population risk, analytical expectations, paired summaries and saved-record reproduction. P1 and P2 provide **248,000 canonical fitted-student records**; the feature appendix checks the linear mechanism on real pretrained representations.

The source is Agus Sudjianto and Denis Burakov, [*Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models*, arXiv:2610.26797v1](https://arxiv.org/html/2610.26797v1). P1 is a two-dimensional special case of §4.2; P2 reproduces one Table 1 Gaussian setting. The [claim map](CLAIM_MAP.md) links assumptions, equations, implementation files and checks. Values below are rounded for reading; [facts.json](facts.json) and [raw closeout summary](results/closure_posthoc_01/summary.json) preserve full precision.

## Models and independent calculations

In P1, independent standard Gaussian variables X1 and U generate correlated input X2. The true signal and the student's input are X1. Writing the teacher as `T=(1+a)X1+bU` separates error predictable from the student's input from its Gaussian residual. For fixed teacher/representation, unregularized OLS without an intercept, and independent IID test inputs,

```
E_train R_truth = a² + b²/(n−2)
E_train R_teacher = b²(1 + 1/(n−2)), n>2.
```

The implementation compares the actual fitted coefficient's risk with this expectation through separate scalar, quadratic-form and solver paths. The derivation uses conditional moments and the inverse chi-square moment; nondegenerate risk variance requires n>4. An unseen coordinate can still be predictable through input correlation, so an unobserved coordinate is not automatically a zero-projection error.

P2 uses d=5, r=3, n=27, lambda=0.45, identity input covariance and a fixed representation retaining three coordinates. The four methods use either full or reduced input, and outcome, teacher or mixed targets. Correlated target noises and representation loss are retained. Its risk is against the true signal, excluding outcome observation noise. General matrix projection calculations use `solve`, while OLS uses least squares rather than a normal-equation inverse. Independent coordinate/Fraction calculations check the analytical expectations; sample signed cross terms remain in the fitted-risk decomposition.

## P1 and P2 results

| P1 condition | Analytical expectation | Recorded mean |
|---|---:|---:|
| rho0 / n30, visible truth risk | 0.250000 | 0.250000 |
| Same condition, teacher agreement | 0 | approximately 1.5e−31 |
| rho0 / n30, hidden truth risk | 0.008929 | 0.008461 |
| rho0.9 / n1000, hidden truth risk | 0.202548 | 0.202550 |

P1 contains 84 settings × 2,000 repetitions: 168,000 canonical fit rows on 62,000 unique datasets. Teacher agreement here is an independent-test population risk, not a training residual. The visible-error example shows why agreement with a teacher alone does not establish truth accuracy.

| P2 method | Input / target | Expected truth risk | Recorded mean | MC-SE |
|---|---|---:|---:|---:|
| FO | Full / outcome | 0.309524 | 0.310155 | 0.001637 |
| FT | Full / teacher | 0.404467 | 0.403867 | 0.001518 |
| RO | Reduced / outcome | 0.260000 | 0.259864 | 0.001153 |
| RM | Reduced / mixed | 0.206875 | 0.207679 | 0.000791 |

All four methods share 20,000 training datasets, yielding 80,000 fit rows. RM's lowest recorded mean concerns this fixed setting, not a general advantage of distillation or reduced input. The historical verifier and hosted CPU CI cover 36 tests and all 248,000 canonical rows. The closeout reused their receipts and identities rather than repeating full sweeps. Sampling intervals and the four-SE investigation rule are described in the historical records, not treated as numerical proofs of the theorem.

## Real pretrained features and the closeout measurement

The feature work uses one frozen SmolLM2-135M checkpoint, FP32 final-layer last-nonpadding-token representations and a linear head. PCA and scaling are fitted only on calibration features. The earlier PCA16 task evaluation stopped; a post-hoc, cache-only diagnosis then compared a fixed set of wider readouts:

| OLS feature space | Train-only CV MSE | Development validation MSE |
|---|---:|---:|
| PCA16 | 0.299546 | 0.300690 |
| PCA64 | 0.182789 | 0.175955 |
| Full hidden | 0.048071 | 0.044506 |

This supports a PCA16 bottleneck for that training sample and development set. PCA preserves input variance, which need not preserve label-predictive directions. A saturated full-hidden in-sample error projection is not evidence of heldout predictability; the fixed calibration-to-validation probes did not establish improved generalization of the old low error.

The subsequent PCA64 experiment's independent qualification had gold MSE 0.183002 and projection gap 0.931817, but high teacher-error MSE 0.233394 missed the registered 0.2375–0.2625 range; low error MSE was 0.262229. The experiment stopped without main/final execution. The original F2 task stop and F2R qualification stop remain unchanged.

The closeout uses that already-accessed qualification set as post-hoc evaluation data. It fits four actual targets `y*±e_high` and `y*±e_low` on the same stored PCA64 training rows, with the frozen original errors and gold coefficient. There is one training dataset, 2,048 train rows and 2,048 evaluation rows. All four added OLS fits are valid; no teacher target was rescaled to force equal heldout errors.

| Error direction | Truth risk R, ± average | Agreement A, ± average | Actual error E | Risk increase Delta | Transfer ratio G |
|---|---:|---:|---:|---:|---:|
| High | 0.4164 | 0.1830 | 0.2334 | 0.2334 | 1.0000 |
| Low | 0.1992 | 0.4456 | 0.2622 | 0.0162 | 0.0619 |

*Post-hoc results for one training dataset and reused qualification data. Gold truth risk is 0.1830; R and A average the actual plus/minus fits. Full-precision individual-arm risks and signed cross terms remain in the saved records.*

Here `Delta=(R_plus+R_minus)/2−R_gold` and `G=Delta/mean(e_eval²)`. G is a ratio of mean energies, not accuracy, a probability or a population projection fraction. It uses actual evaluation error energy rather than nominal 0.25. No new statistical test or multi-seed confidence interval was introduced.

For a fixed linear learner, writing `a=f_gold−y*` and `d=f_arm−f_gold` gives

```
R_arm−R_gold = mean(d²) + 2 mean(a*d).
f_plus/minus = f_gold +/- K e_train
Delta = mean((K e_train)²).
```

Each signed cross term is retained; only the sign average cancels it. High error lies in the fixed PCA64 linear span and the train design has full rank, so G_high≈1 is a check of known OLS linearity. Low error is a calibration residual, not an assertion of population orthogonality or independence. Its transfer ratio was measured. The registered equal-error H1 effect/interval and H2/H3 remain unrun; these descriptive G results do not replace them.

## Reproduce and inspect the evidence

The appendix contains the small PCA64 inputs, labels/errors, IDs, original metadata, fitted coefficients and row-level predictions. It needs no weights or GPU. In an extracted package root, [REPRODUCTION.md](REPRODUCTION.md) gives the exact installation, verification and fit-free summary commands. The original paper-record path is documented [separately](https://github.com/Munsik-Kim/teacher-error-geometry-lab/blob/v0.1.0/docs/REPRODUCTION.md).

Four primary target fits use FP64 SVD/lstsq. Independent reduced-QR gold/error-only calculations, serialized-coefficient prediction replay and residual-versus-quadratic MSE comparison passed 53 checks; nine separate synthetic-array tests passed. The numeric tolerance was fixed before these results. Reaggregation performs zero fits. Exact call accounting, timings, source identities, historical bug fixes, test exposure and preservation receipts are collected in [PROVENANCE_AND_HISTORY.md](PROVENANCE_AND_HISTORY.md) and [facts.json](facts.json).

The completed scope is Gaussian linear-theory partial reproduction and a fixed pretrained-feature linear mechanism appendix. AI assisted implementation and internal checking; the records distinguish this from human independent review. Assumptions, unrun extensions and the original stop history are consolidated in the provenance document rather than reinterpreted as successful experiments.
