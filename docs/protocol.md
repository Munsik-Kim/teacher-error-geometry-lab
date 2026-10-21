# Public protocol

This public candidate preserves both historical phases. P1 is the controlled two-dimensional B_A=0 special case; P2 fixes d=5,r=3,n=27,Sigma=I,lambda=.45,R=20,000. The original frozen protocols and source manifests remain in the [evidence archive](EVIDENCE.md), together with preflight and run histories.

All scientific generator, solver, scorer, theory, summary and plotting modules remain byte-identical to the completed research tree. P3 changes only installation, root/path discovery, CLI wiring, optional legacy preservation metadata and public verification tools. See [portability changes](PORTABILITY.md).

Use FP64, no intercept/regularization/jitter/whitening, np.linalg.lstsq; OLS receives assigned designs/targets only. P1/P2 pairing and SeedSequence streams follow their frozen configs. No new full sweep, n/lambda choice or Monte Carlo experiment is part of P3. Unit tests use small fixtures; recorded verification replays only historical preselected keys.

Risk is population error against the true signal, not training MSE or noisy Y prediction. Deterministic checks retain atol=1e-10,rtol=1e-9. Stochastic comparisons use paired differences, sample SD ddof=1 and the original 4SE investigation rule. Same-data reanalysis/replay is not additional independent evidence. The P1 post-observation structural correction is retained in [history](HISTORY.md).

[Results](RESULTS.md) · [Theory](theory.md) · [P2 protocol](P2_PROTOCOL.md) · [Reproduction](REPRODUCTION.md)
