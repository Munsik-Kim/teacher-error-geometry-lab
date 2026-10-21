# Recorded results

[한국어](RESULTS.ko.md) · [Full-precision shared facts](../data/results.json)

P1: 84 settings, 2,000 repetitions per setting, 168,000 fitted-student rows on 62,000 unique datasets; no failed or missing rows. The two historical full invocations made 336,000 solver calls on the same samples, not 4,000 independent repetitions. Final implementation verification is PASS and statistical comparison is NO_4SE_FLAGS.

At rho=0,n=30 the visible teacher has truth risk 0.25 and teacher-agreement MSE 1.4976031247555146e-31. The hidden teacher gives measured truth risk 0.00846083145254395 against expectation 0.008928571428571428. Projection-zero errors have a=0 even for correlated inputs. At rho=.9,n=1000 hidden risk is 0.20255038730293556, expectation 0.20254759519038076 and population floor 0.2025. This is the fixed-representation B_A=0 special case.

P2: n=27,lambda=.45; each method has R_planned=R_rows=N_valid=20,000, N_failed=N_missing=0. All four share the same 20,000 datasets; four groups and six paired comparisons are dependent. FO uses full X/Y, FT full X/T, RO reduced Z/Y, RM reduced Z/((1-lambda)Y+lambda T).

|Method|Theory|Measured mean|MC-SE|z|
|---|---:|---:|---:|---:|
|FO|0.30952380952380953|0.31015494835202323|0.0016367640176310073|0.3856016025616098|
|FT|0.40446666666666664|0.4038669197012249|0.0015184219667590693|-0.3949804326934514|
|RO|0.26|0.2598640016423335|0.0011526614747491982|-0.11798638251193809|
|RM|0.20687515217391308|0.20767895944476805|0.0007907319008408488|1.0165357815970422|

RM has the smallest observed mean in this one setting. The methods differ in both input information and targets, so this does not establish a general distillation or compression advantage. The fixed lambda result is distinct from minimizing risk over lambda.

The RM expected terms are .08 representation loss + .01524825 projected teacher bias + 0 expected cross term + .11162690217391306 estimation variation. The actual MC components are .08000000000000004 + .015248250000000014 + .00041106984100782904 + .1120196396037602, giving truth risk .20767895944476805 up to floating-point summation. The signed cross term was retained. P2 introduces B_A=.08 and residual coupling S_A+omega=.41. FT's observed cross mean is -.0005603832093237588.

![Four methods, fixed Gaussian anchor](../recorded/p2/figures/01_four_methods_truth_risk.png)

![RM decomposition, signed sample cross term retained](../recorded/p2/figures/02_rm_decomposition.png)

P1/P2 use pointwise mean +/-1.96 MC-SE intervals with sample SD (ddof=1). The 4SE rule is an investigation threshold, not simultaneous coverage or a numerical proof of theorems. Some P1 95% intervals exclude theory; the maximum marginal absolute z is 3.027129256419131. P2 has no group/pair/component 4SE flags; maximum absolute paired z is 1.125915430907219. Deterministic identities use fixed atol=1e-10,rtol=1e-9 instead of division by tiny SE. See [history](HISTORY.md) for P1's post-observation diagnostic correction.

Recorded runner wall times: P1 final 9.23998189899794 s, two historical full runs 18.40799369999877 s; P2 runner 5.903453523002099 s and enclosing full CLI 6.07 s. The CLI includes its runner: do not add them. These exclude environment installation, validation and figure reproduction; they are not hardware-normalized or serving benchmarks. See [original execution receipts](../recorded/p1/provenance/execution.json), [P2 receipt](../recorded/p2/provenance/execution.json) and [P3 costs](P3_REPORT.md).

[Historical reports and complete raw restoration](EVIDENCE.md) preserve the original reports byte-for-byte. Earlier NOT_RUN states describe their historical time. Current P3 preparation is LOCAL_ONLY; no new science or full experiment was run.
