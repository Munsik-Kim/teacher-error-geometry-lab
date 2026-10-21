# P2 fixed Gaussian anchor

Table 1 anchor: d=5,r=3,n=27,A=[I3,0],Sigma=I5; beta=(.9,-.6,.5,.2,-.2), delta=(.22,-.13,.10,.35,-.20). sigma2=1.3,tau2=.7 are variances; omega=.22 is covariance. The Gaussian noise factor has determinant .8616>0. X and noises are independent across cases; sample covariance is never forced.

Each of 20,000 datasets supplies X,Y=X beta+epsilon,T=X(beta+delta)+xi,V=.55Y+.45T to four methods. FO/FT use full X with Y/T; RO/RM use Z=XA^T with Y/V. Four groups and six unordered pairs share each replicate. See [fixed config](../configs/p2_paper_anchor.json), [original freeze](../recorded/p2/provenance/freeze.json) and [source correspondence](P2_SOURCE_MAP.ko.md).

Actual risk uses the fitted 5D quadratic form, lifting reduced coefficients with A^T. Expected risk is calculated separately. RM sample decomposition retains 2lambda(w_gamma-w_beta)^T C h, whose expectation alone is zero. No oracle correction enters the solver. Independent Fraction constants, QR and test-MC checks are in the original 12 P2 tests/preflight; P3 retains those tests and runs no full experiment.

Root seed 2026102102 is separate from P1 and the source paper's PyTorch seed 20261016. NumPy backend samples are not a bitwise replication. Full records, source and original reports are restored using the [evidence guide](EVIDENCE.md).
