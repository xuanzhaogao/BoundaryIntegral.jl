# Conditioning of the dielectric BIE LHS: edge correction and diagonal preconditioning

**Date:** 2026-07-31
**Status:** both changes landed as defaults; production-scale A/B not yet run
**Reproducer:** `examples/diag_precond_experiment.jl`, driver `examples/run_precond_2x2.sh`

## Summary

Two independent defects were inflating the GMRES iteration count of the interface solve.
Fixing both reduces it by 10–35× on a multi-material benchmark, and the two fixes are
near-multiplicative because they act on different parts of the operator.

1. **Edge correction was off by default** (`correct_edges = false`). This dropped every rim
   (`is_edge`) panel from the near list as both source *and* target, and built no
   touching-pair corrections at all. The result was an operator whose conditioning
   *degrades as the mesh is refined at the edges*, so `N_it` was not mesh-independent —
   contradicting the claim made in `main.tex:418`, `:1205` and `:1212`.
   Now defaults to `true`.
2. **The contrast term `G` was left unscaled.** Unlike a textbook second-kind equation the
   identity term is not `½I`: `A = G + Dᵀ + Cᵀ` with `G = diag(t_P)` constant per interface,
   so the spectrum sits in one cluster per distinct contrast. Right-preconditioning by
   `P = G⁻¹` collapses them onto a single cluster at 1. Now defaults to `true`
   (`precondition = true`).

## Results

Three-material benchmark of §6.1 (10×10×1 slab ε₃=10 on two 10×10×10 cubes ε₁=4, ε₂ swept,
sharing an internal face, in vacuum; Gaussian source inside the slab). p=6, τ_rhs=1e-3,
l_ec=1.25, N=24480, `gmres_rtol=1e-8`, 96 cores on worker7054.

| ε₂ | neither | precond only | edges only | **both** | both vs neither |
|---|---|---|---|---|---|
| 6   | 179 | 128 | 27 | **17** | 10.5× |
| 20  | 266 | 129 | 38 | **18** | 14.8× |
| 60  | 587 | 276 | 45 | **22** | 26.7× |
| 200 | 828 | 401 | 51 | **24** | 34.5× |
| single box (one γ) | 26 | 26 | 12 | **12** | 2.2× |

Wall clock at ε₂=200 (96 cores, setup + solve): 104.6 s → 3.4 s.

**Near-multiplicative.** At ε₂=200, edges alone give 16.2×, the preconditioner alone 2.1×,
and 16.2 × 2.1 = 34 ≈ the measured 34.5. The preconditioner keeps its full factor after the
edge correction: they fix independent defects.

**Mesh independence restored.** With edges corrected, `N_it` is flat in N (26→27 at ε₂=6 and
44→51 at ε₂=200 as N goes 3.6e3→2.4e4). With edges off it grows steeply (32→179, 79→828).

**Contrast penalty largely removed.** Growth across ε₂=6→200: 4.6× (neither) → 3.1×
(precond) → 1.9× (edges) → **1.4× (both)**. §6.1 currently describes this as a
mesh-independent increase in iteration count of roughly 2.4× (16–18 → 35–43); with both
fixes it is ~40%.

**Preconditioner invariance check.** For a single contrast `P` is a scalar multiple of `I`,
and GMRES is invariant under scalar scaling of `(A, F)`, so `N_it` must be *identical*. It is
(9→9, 12→12, 26→26 across resolutions), and σ agrees to 1e-15. This is asserted by the
reproducer and is the cheapest guard against a sign or indexing error in the scaling.

## Why the preconditioner is safe

`γ = (ε_k+ε_k′)/(ε_k−ε_k′)` satisfies `|γ| > 1` for any pair of positive permittivities, so
the scaling entries `|1/t| = |2/γ| < 2`: uniformly bounded, never amplifying. It is applied
as a **right** preconditioner (`N = Diagonal(s)` to `Krylov.block_gmres`) rather than left,
so block GMRES keeps minimizing the true residual `‖F − AΣ‖` and `rtol` keeps its meaning;
Krylov applies `N` to the returned solution, so `Σ` needs no post-scaling. For diagonal `P`,
`AP` and `PA` are similar, so right preconditioning gives the same spectral clustering as
left.

## Open issue: the ∫σ prefactor anomaly

The two accuracy metrics in the test suite disagree about the edge correction, and the
disagreement is *not* a tolerance artifact.

Total induced charge against the exact Gauss's-law value `1 − 1/ε` (3×3×1 box, ε=4, p=6):

| l_ec | N | err (edges off) | err (edges on) | rate off | rate on |
|---|---|---|---|---|---|
| 0.400 | 10656 | 1.53e-3 | 4.36e-3 | — | — |
| 0.200 | 28800 | 8.97e-4 | 2.49e-3 | 0.77 | 0.81 |
| 0.100 | 67680 | 5.22e-4 | 1.42e-3 | 0.78 | 0.81 |
| 0.050 | 148032 | 3.02e-4 | 8.14e-4 | 0.79 | 0.81 |

Both converge at the rate ≈0.8 predicted by the unresolved-strip estimate
`O(h_min^(β+1))` with β≈−0.2, so the edge correction changes only the constant — and
edges-off has a ~2.9× *smaller* constant at every level. This is not one-mesh cancellation,
and not an under-resolved touching quadrature (the flux error is flat in `adaptive_atol` from
1e-6 down to 1e-12). Meanwhile edges-*on* is 3.5× better on the near-corner pointwise
potential (`test/solver/dielectric_box3d.jl:279`).

Untested hypothesis: the plain GL rule's error on rim pairs has opposite signs on the two
faces meeting at an edge, cancelling in the signed integral `∫σ` but not pointwise.

This matters because Fig. 3(b) validates against exactly this `E_σ` metric. It does *not*
appear to matter for `V`: an earlier measurement put the difference at ~2–4e-6 relative,
well below the 1e-3 target. **The default flip is therefore justified on cost, not on
accuracy.**

## Production-scale evidence (indicative, not conclusive)

The §6.3 lattice campaign (`/mnt/ceph/users/xgao1/four_index/lattice_10x10_het3x`, 198
batches, run with `correct_edges = false`) records `stats["niter"]` per batch:

| batch | 1 | 5 | 20 | 40 | 60 | 90 | 120 | 150 | 180 | 198 |
|---|---|---|---|---|---|---|---|---|---|---|
| niter | 67 | 62 | 66 | 61 | 61 | 72 | 64 | 64 | 79 | 86 |
| K | 13 | 17 | 15 | 17 | 17 | 10 | 15 | 17 | 5 | 1 |

All at dof ≈ 651k. Compare Table (multirhs), which reports 29–36 for the §6.2 benchmark at
2.5× *more* unknowns. The K-dependence matches the paper's block-GMRES claim (K=1 → 86,
K=17 → 61). The geometries differ (270 Å cubes, thin 43×44×9 Å slab vs 90 Å cubes), so this
is *not* proof of an edge-correction tax — but the solve stage that consumed 84% of 39
node-hours ran at 61–86 iterations, where the corrected operator should need far fewer.

**Decisive test, not yet run:** each `batch_NNNN.jls` stores the interface and the source
densities, so the operator and RHS can be rebuilt from batch 1 and solved four ways at the
real N=651k, K=13 without re-running the pipeline — roughly 20 min on a 96-core node.

## Code changes

| file | change |
|---|---|
| `src/kernel/laplace3d_near.jl:394,420` | `laplace3d_{DT,D}_fmm3d_corrected` → `correct_edges = true` |
| `src/kernel/laplace3d_near.jl:234` | comment: why `build_neighbor_list` itself stays `false` |
| `src/solver/dielectric_box3d.jl:53` | `lhs_dielectric_box3d_fmm3d_corrected` → `true` |
| `src/solver/multi_rhs.jl` | `dielectric_diagonal_scaling`; `precondition = true`, `correct_edges = true` on `solve_dielectric_box3d_block` |
| `src/solver/lattice_batch.jl:291` | both flags threaded through `solve_dielectric_lattice_batch`, defaults `true` |
| `src/BoundaryIntegral.jl:64` | export `dielectric_diagonal_scaling` |
| `examples/diag_precond_experiment.jl`, `examples/run_precond_2x2.sh` | reproducer |

`build_neighbor_list` deliberately keeps `correct_edges = false` as its own default:
`laplace3d_DT_fmm3d_corrected_hcubature` calls it without the flag and consumes only the
`upsample` dict, so flipping the primitive would make that path silently discard the
`adaptive` (touching) pairs. Every operator-level caller now passes `true` explicitly.
Making that path correct touching pairs too is a loose end.

Verification: full suite (`BI_RUN_FULL_TESTS=1`) 15750/15750 with `correct_edges = true`;
rerun after the `precondition` flip. Every test that cares passes the flags explicitly.

## Implications for the paper

1. §6.1's flat-count and edge-correction claims describe the corrected operator; §6.3's
   numbers were produced without it. Whichever way the production A/B lands, the two
   sections need to state which configuration they used.
2. If the tax is real at scale, the §6.3 timing table (solve = 3 h 14 m, 84% of 39
   node-hours) is a measurement of the uncorrected operator and would improve substantially.
3. The contrast study (Table (contrast)) is worth rerunning with both fixes: the story
   changes from "iteration count roughly doubles toward the conductor limit" to "grows ~40%".
4. Neither fix changes `V` materially (~1e-6), so published tensor values stand.
