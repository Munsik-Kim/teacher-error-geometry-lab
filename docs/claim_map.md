# Claim map / 원문 대응표

Mathematical source: Agus Sudjianto and Denis Burakov, *Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models*, [arXiv:2610.26797v1](https://arxiv.org/html/2610.26797v1), 2026-10-19. P3 uses the completed P0/P1/P2 source-access records; it does not claim a new independent paper review.

|원문 위치 / source|가정·구현 대응 / correspondence|검증 / evidence|
|---|---|---|
|§3.1 Lemma 3.1|Gaussian full outcome OLS; FO target Y, signal X beta|matrix expectation, independent Fraction constant 13/42, saved 20,000 FO risks|
|§3.2 Proposition 3.2|full teacher regression FT, gamma=beta+delta, xi variance tau2|B=.2378 plus tau2*d/(n-d-1), quadratic fitted risks|
|§4.1 Proposition 4.1|fixed A=[I3,0], representation loss B_A=.08; RO learns Y|projected matrix path and coordinate path, expected .26|
|§4.2 Theorem 4.3|RM fixed mixture lambda=.45; residual coupling S_A+omega=.41|D_A=.0753,G_A=.4625,S_A=.19; Fraction 9516257/46000000; signed sample cross identity|
|§4.2 Theorem 4.3 / Corollary 4.5|P1 r=1,lambda=1,zero extra noise,B_A=0,D_A=a²,G_A=b²|84 controlled settings, analytic risk and paired raw summaries|
|§7.1 Table 1|P2 single d5/r3/n27 Gaussian setting|four methods ×20,000, fixed lambda=.45, six paired differences|

P1 is our controlled special case, not the paper's original full simulation. P2 is expectation reproduction for a single Gaussian setting, not bitwise sample/MC-mean reproduction. NumPy/lstsq and Matplotlib are upstream; generation, orchestration, evaluation and reproduction tools are independently implemented. No author code was accessed. AI-assisted internal checking is not human peer review or another-machine execution.

The scope excludes n/lambda sweeps, optimal-lambda or full-threshold validation, non-Gaussian data, train/test correlation shifts, probit, CFPB, learned representations and actual LLMs. No novelty or general distillation/compression/quantization/serving improvement is claimed.

[Theory](theory.md) · [P2 variables](P2_SOURCE_MAP.ko.md) · [Source-access boundary](source_access.md) · [Results](RESULTS.md) · [Evidence restoration](EVIDENCE.md)
