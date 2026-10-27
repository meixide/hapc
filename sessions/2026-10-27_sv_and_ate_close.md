# HAPC development closure — 27 October 2026

## Scope and navigation

Scientific development during 23–27 October; closure 27 October. This record integrates the ATE-TMLE baseline and the subsequent norm1/SV algorithm audit. It is based on the development conversation, saved run manifests and reports. Historical runs are preserved; they were not all rerun at closure. The public package contains code and synthetic experiments only.

Start with this record, [SV-ball report](../examples/sv_ball/README.md), [previous boundary audit](../examples/sv_stress/README.md), and [ATE-TMLE report](../examples/ate_tmle/README.md). Each report links configuration, code snapshots, environment and per-case results. Snapshots deliberately preserve the source that produced each run, including shared engine changes present at the time; they are historical inputs, not another current API.

## Decisions and mathematical distinction

The research priority is one estimator in method, theory and simulations. In canonical HAL atoms, the coefficient L1 norm gives nonconstant sectional variation; the anchored intercept is additional. With redundant atoms the coefficient norm is an upper bound unless equivalent atoms have been aggregated. PC coefficients alpha are not original HAL coefficients: beta=V alpha.

L1(alpha)<=C alone does not uniformly bound L1(beta) when the dictionary grows. For orthonormal V, L1(beta)<=sqrt(N)*L2(alpha)<=sqrt(N*L1(alpha)*Linfty(alpha)). L1(alpha)<=C and Linfty(alpha)<=C/n suffice if N=O(n). Ordinary norm1 CV does not impose the latter condition. In orthogonal Gaussian regression alpha_j=soft(s_j,lambda)/q_j; lambda>=max_j(|s_j|-q_j*C/n)_+ imposes the box, but changes selection. A binary repeated-atom example needs a PC coordinate of order n^(-1/2): blindly imposing a fixed C/n box destroys approximation. L2(alpha)<=M/sqrt(N) is a weaker sufficient route, still needing approximation analysis. No general automatic-CV guarantee was established.

The requested original equality algorithm was audited first. After its limitations were demonstrated, the user expressly accepted replacing it by minimisation over L1(V alpha)<=C. The existing initializer still determines C(lambda); this closure does not endorse the current CV/radius selection. Interior solutions are allowed, so the statistical estimator changes when the constraint is slack. The boundary comparator remains explicitly callable. A stress test comparing the number 1 in three different norms measures norm control, not equal statistical complexity or predictive superiority.

## Implementation and evidence

The boundary audit repaired leaked radius at beta zeros, risk-increasing steps, stale-gradient final selection and inconsistent Python PC decompositions; it added feasible histories, safeguarded logistic intercepts and relative numerical-zero handling. It also demonstrated a structural limitation: multiplicative updates lock alpha zeros. For Z=V=I, y=(0,2), alpha0=(1,0), the score can vanish at risk 2.5 without even a local constrained minimum.

The accepted ball solver combines ADMM with exact L1 projection and, when needed, refinement of the same convex problem using an epigraph barrier. Feasible incumbents and valid dual lower bounds are retained. The counterexample now reaches (0,1), risk 0.5. Convergence requires a numerical primal-dual gap <=tol*(1+abs(risk)), plus rank/residual checks. It is a floating-point certificate, not a formal verified proof. A stalled or iteration-limited fit is not certified. R/Python single and all-fold CV outputs expose status, radius and gap. Numerical-rank truncation removes roundoff PCs. The detailed report derives the dual and intercept bounds and describes API compatibility changes.

Final saved run: `examples/sv_ball/results/2026-10-27-ball-v1/`. All 44 fits meet scaled tolerance 1e-8; maximum scaled gap 9.981e-9 and positive budget excess 1.78e-15. All risks are lower than matched boundary fits in this run. Ten independent SLSQP epigraph references agree in risk within 8.68e-9. Eighteen Python tests, the new R native/public/CV checks and 77 existing R binomial checks pass. These are selected relevant suites, not a full package check. Builds used isolated libraries; no user installation was replaced.

The ATE baseline is separate: `hapc.ate.tmle` independently implements standard point-treatment ATE targeting with HAPC initial fits and EIF intervals. It supports binary/continuous complete data, not ATT/ATC, missingness or proximal targeting. No tmle/tmle3 code was copied and no dependency on them was added. In 160 HAPC synthetic fits (40 replicates per outcome/size, n=100/300), TMLE and one-step are very similar; coverage 97.5–100% is exploratory with substantial Monte Carlo uncertainty. Twelve glmnet partial-path warnings were retained. No superiority or calibrated coverage is established. All 26 ATE test expectations were rerun successfully against the final isolated R build at closure, with no warnings/failures in that test run. This does not rerun the simulation or resolve its 12 warnings.

## Pending work, in order

1. Audit CV and radius construction before a larger campaign. Gaussian SV currently builds a global covariate basis and uses nonorthogonal row subsets for fold initialization; binomial SV also keeps the global covariate basis. Define the intended fold estimator, training-size scaling and refit rule before changing them.
2. Verify approximation at the retained PC rank and selected radius, including anchored intercept control. Fully constant designs, explicit HAL materialisation at scale and arbitrary rank-deficient penalty maps are not claimed validated domains.
3. Align the scientific manuscript's estimator, algorithm, KKT/score equations and numerical evidence with the ball. Numerical optimisation certificates do not establish statistical rates or interval coverage.
4. Predefine a statistical simulation plan with paired seeds/folds, failures and Monte Carlo error; do not relabel older ridge or PC-L1 runs as SV-ball results.
5. For the separate ATE baseline, investigate convergence warnings and compare upstream targeting with identical externally supplied nuisance predictions before claiming package equivalence. More complex estimands remain later work.

No package release, external submission or contact with collaborators is part of this closure. Commit/push of the reviewed development changes is part of the requested closure; Git history provides the resulting version. Earlier reports saying “no commit/push” describe their historical stage, not this closure.

## Reusable workflow assessment

Reuse coherence auditing and reproducibility checks: write the actual constrained objective, distinguish feasibility/stationarity/global optimality, retain failed cases and trace each table to an exact source. A possible minimal addition to an existing numerical-audit procedure is to require an independent reference and explicit optimisation certificate for a claimed convex solver. This conversation supports that proposal but is not evidence of recurring independent incidents. No new skill was created or deployed.

Closure checks: current edited sources pass Git whitespace checks. Historical snapshots, compiler logs and CSV line endings are retained byte-for-byte; their whitespace warnings are not repaired because doing so would invalidate saved source hashes. Both saved SV source manifests were rechecked with zero mismatches.
