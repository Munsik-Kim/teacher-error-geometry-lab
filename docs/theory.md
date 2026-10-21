> Public reading copy of the completed P1 derivation. Paper-reading statements below describe the historical P1 work. The byte-original document restores as docs/theory.md.

# Direct derivation: two-dimensional Gaussian P1

Let independent training pairs have X1,U ~ N(0,1), X2=rho X1+s U,
s=sqrt(1-rho^2), |rho|<1. Thus E[X1 X2]=rho and Var(X2)=1 by
construction. Sample covariances fluctuate; we never correct them. The student
observes Z=X1 and its target T, with no intercept or regularization.

The truth is X1 and the fixed teacher is T=(1+delta1)X1+delta2 X2.
Define a=delta1+rho delta2 and b=delta2 s. Then T=(1+a)X1+bU.
The one-column OLS identity, derived by differentiating the squared residual
objective, is

    w_hat = sum(X1 T)/sum(X1^2)
          = 1+a + b sum(X1 U)/sum(X1^2).

The implementation fits with np.linalg.lstsq, not this identity. The identity is
an independent numerical check and is never an input to the learner.

Write S=sum(X1^2) and Q=sum(X1 U)/S. Conditional on all X1 values,
sum(X1 U) ~ N(0,S), so

    E[w_hat | X1] = 1+a,
    Var(w_hat | X1) = b^2/S.

S ~ chi-square(n). Integrating its gamma density gives
E[1/S]=1/(n-2), n>2: the gamma integral changes shape from n/2 to n/2-1.
Therefore E[Q]=0 and E[Q^2]=1/(n-2). These expectations average over
newly generated training data, not a large test sample for one fit.

For a fixed w, a new independent X1 has variance one. Consequently

    R_truth(w) = (w-1)^2.

For a fresh independent pair (X1,U), student minus teacher is
(w-1-a)X1-bU. Independence gives

    R_teacher(w) = (w-1-a)^2+b^2
                 = h^T Sigma h, h=(w-1-delta1,-delta2).

The raw risk evaluator uses the quadratic form. The conditional expansion is a
separate check; the expectation function does not call the fitted evaluator.
Taking training-data expectations yields

    E[R_truth]   = a^2+b^2/(n-2),
    E[R_teacher] = b^2*(1+1/(n-2)),             n>2.

Neither is the training residual MSE. For nondegenerate b, E[Q^4] is
3/((n-2)(n-4)), so finite variance of the risk Monte Carlo estimator requires
n>4. In particular, with v=b^2/(n-2),

    Var(R_teacher) = 3b^4/((n-2)(n-4))-v^2,
    Var(R_truth)   = Var(R_teacher)+4a^2 v.

The full sweep has n>=10. Reported MC-SE uses the sample SD, not these optional
theoretical SD diagnostics. If b=0 the risks are deterministic; FP64 roundoff is
checked with the frozen tolerance rather than standardized by a tiny SE.

For paired teachers on the same dataset, Q is shared. Their teacher-risk
difference is (b_left^2-b_right^2)*(1+Q^2); it is deterministic zero when their
absolute hidden loadings agree. Their truth-risk difference has stochastic
coefficients 2*(a_left*b_left-a_right*b_right) and b_left^2-b_right^2.
Both must vanish for a deterministic truth-risk difference. At rho=0 the
normalized angles pi/4 and3pi/4 have the same hidden loading exactly in real
arithmetic, so their paired teacher-risk difference is tested with the fixed
FP64 tolerance. It is not divided by a roundoff-sized empirical SE.

## Geometry and normalized angle

Population teacher error is delta^T Sigma delta=a^2+b^2. Every planned teacher
has this quantity c^2, even when Euclidean ||delta|| changes. For c>0, set
eta=a^2/c^2. Then

    E[R_truth]/c^2 = eta+(1-eta)/(n-2).

The implementation records eta=None at zero error and returns risk zero in the
c=0 curve branch; it does not divide by zero. For projection-zero teachers,
delta=c*(-rho,1)/sqrt(1-rho^2), a=0 and b=c. Their expected truth risk is
c^2/(n-2) for every rho. At rho=0 this teacher and axis_hidden coincide.
For the normalized-angle family v=(cos(theta),sin(theta)), use
delta=c*v/sqrt(v^T Sigma v); theta labels a Euclidean direction, not eta.

## Paper correspondence

Directly read [Sudjianto and Burakov, arXiv:2610.26797v1, §4.2](https://arxiv.org/html/2610.26797v1#S4.SS2),
Theorem 4.3, equations (11)-(12), and Corollary 4.5, equation (13), plus the
Gaussian/fixed-map assumptions in §3 and §4.1. Correspondence:

| Paper object | P1 value |
| --- | --- |
| d, Sigma, fixed full-row-rank A | 2, [[1,rho],[rho,1]], [1,0] |
| beta, gamma | (1,0), beta+delta |
| r, lambda | 1, 1 (teacher targets only) |
| sigma^2, tau^2, omega | 0, 0, 0 (no extra target noise) |
| C_A, w_*, u_beta | 1, 1, (0,0) |
| w_gamma, u_gamma | 1+a, (-rho*delta2,delta2) |
| B_A, D_A, G_A, S_A | 0, a^2, b^2, 0 |
| A Sigma delta | a |

Substituting these quantities into Theorem 4.3 gives the truth-risk expectation.
Corollary 4.5 identifies D_A as the projected teacher error and its zero condition
as a=0. The teacher-agreement identity above is derived directly here.
Assumptions used: independent Gaussian design, positive definite covariance,
fixed representation and teacher coefficients before sampling, an independent
test input of the same distribution, no fitted intercept, and n>r+1.
This is a special-case implementation, not a reproduction of the full paper,
its actual LLMs, probit results, or CFPB experiments.

## Separate gold control and explanatory prediction

A student given gold targets X1 learns w=1 and truth risk zero in this noiseless
model. This is a separate P0 control; gold is never sent to a teacher-training fit.
For c=.5,rho=0,n=30, the **analytic prediction** is visible truth=.25,
teacher=0; hidden expected truth=.25/28=.008928571428571428 and expected
teacher=.25*(1+1/28)=.25892857142857145. These numbers are predictions,
not newly measured Monte Carlo results. Teacher agreement alone does not certify
truth accuracy in this example.

## P2 projected Gaussian anchor

For the fixed full-row-rank A and positive-definite Sigma, define C=A Sigma A^T, w_beta=solve(C,A Sigma beta), w_gamma=solve(C,A Sigma gamma), u_beta=beta-A^T w_beta and u_gamma=gamma-A^T w_gamma. The independent projection path yields B=delta^T Sigma delta=.2378, B_A=u_beta^T Sigma u_beta=.08, D_A=(w_gamma-w_beta)^T C(w_gamma-w_beta)=.0753, G_A=u_gamma^T Sigma u_gamma=.4625 and S_A=u_beta^T Sigma u_gamma=.19.

With independent Gaussian cases and n>d+1 (full) or n>r+1 (reduced), inverse-Wishart expectations give

    E[R_FO] = sigma2*d/(n-d-1),
    E[R_FT] = B + tau2*d/(n-d-1),
    E[R_RO] = B_A + r*(B_A+sigma2)/(n-r-1),
    E[R_RM] = B_A + lambda^2*D_A + r/(n-r-1)*[
        (1-lambda)^2*(B_A+sigma2) + lambda^2*(G_A+tau2)
        + 2*lambda*(1-lambda)*(S_A+omega)].

These are expectations over new training data. FO/FT actual risk is (theta_hat-beta)^T Sigma(theta_hat-beta); RO/RM lift coefficients with A^T first. They exclude the observation-noise variance. The expectation function never replaces the actual fitted risk.

For each fitted RM, h=w_hat-[(1-lambda)w_beta+lambda*w_gamma] gives

    R_RM = B_A + lambda^2*D_A
           + 2*lambda*(w_gamma-w_beta)^T C h + h^T C h.

The cross term retains its sample sign; only its training-data expectation is zero. Outcome/teacher residual covariance is S_A+omega=.41, where omega is noise covariance, not correlation. At n=27,lambda=.45 the four expected values independently reduce to 13/42,6067/15000,13/50,9516257/46000000. No optimal-lambda or n-threshold sweep is checked. [P2 correspondence](P2_SOURCE_MAP.ko.md).
