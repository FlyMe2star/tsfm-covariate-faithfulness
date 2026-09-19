# P1-SHAPE outline contract

Status: owner-approved direction; results remain placeholders

## Provisional evidence story

1. **Motivation:** covariate-aware forecast accuracy and coarse response checks do not
   determine whether the response is correctly distributed over the horizon.
2. **Distinction:** P1 audits a known-future exogenous path, not a univariate generator
   parameter, optimized counterfactual explanation, or internal representation.
3. **Protocol:** four untouched structural response families, paired replay, and a
   signed metric evaluated at temporal widths 1/2/4/8.
4. **Construct validity:** analytic broken-response fixtures show what each metric can
   and cannot detect before any TSFM output exists.
5. **Behavioral evidence:** complete two-backbone by four-mechanism matrix plus
   transparent fitted references and matched forecast skill.
6. **Bounded conclusion:** state either the verified hidden-distortion scope or a null
   result; never generalize beyond the tested interfaces and mechanisms.

## Planned sections

1. Introduction
2. Related Work
3. Covariate-Response Shape Formulation
4. CovIntervene-SHAPE Protocol
5. Construct Validation
6. Experimental Protocol
7. Results
8. Analysis and Limitations
9. Conclusion

## Required figures and tables

- Figure 1: same direction and total response, different horizon-wise shapes.
- Figure 2: four untouched mechanisms and paired worlds.
- Figure 3: resolution curves `D_1, D_2, D_4, D_8` with bootstrap intervals.
- Figure 4: oracle versus predicted response trajectories for registered representative
  cells selected by a fixed rule, not visual appeal.
- Table 1: literature boundary and audited interfaces.
- Table 2: construct-fixture selectivity matrix.
- Table 3: complete eight-cell primary matrix.
- Table 4: transparent references, forecast skill, and efficiency.

## Claim-writing rules

- Before verification, use future or hypothesis language only.
- Never describe P0 as a failure of the models; it was a failure of the original paper
  hypothesis under its frozen scope.
- If P1 passes, say “for the tested checkpoints and mechanisms,” not “TSFMs generally.”
- If P1 fails, archive the null result and stop; do not promote a diagnostic metric into
  the primary endpoint.
- Real-data sensitivity, if later added, is observational and cannot validate the
  structural-response claim.
