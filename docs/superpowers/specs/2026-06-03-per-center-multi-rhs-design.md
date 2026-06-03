# Per-Center Multi-RHS Batched BIE Solve

**Date:** 2026-06-03
**Branch:** `multi_rhs`
**Status:** Design — approved scope, pending spec review

## Motivation

The four-index integral pipeline (see `~/Articles/four_indices_bie`, Eq. 1) evaluates

$$V_{ijkl} = \iint \rho_{ij}(x_1)\,K(x_1,x_2)\,\rho_{kl}(x_2)\,dx_1\,dx_2,$$

with pair densities $\rho_{ij} = \varphi_i\varphi_j^*$. Scaling to larger systems means
evaluating this for *many* orbital pairs. The key locality fact: because basis functions
decay, $\rho_{ij}$ is negligible unless $\varphi_i$ and $\varphi_j$ overlap. So for a fixed
orbital center $i$, only the $O(1)$ neighbors $j \in \mathrm{neigh}(i)$ contribute, and every
$\rho_{ij}$ in that group is **supported in the same small region near atom $i$**.

The BIE operator $A$ (Eq. 13/18 of the paper) depends only on the interface geometry $\Gamma$
and the dielectric contrasts $\gamma$ — **not on the source**. Only the right-hand side
$f = -\partial_{\V n} u_{\mathrm{inc}}$ changes per source. This is the leverage:

- For a co-located group $\{\rho_{ij}\}_{j\in\mathrm{neigh}(i)}$, one shared interface
  discretization resolves all members, so a single matrix $A$ serves the whole group.
- The group becomes a **multi-RHS block**: solve $A\,\Sigma = F$, $F = [f_{i1}\,\cdots\,f_{iK}]$,
  with one shared Krylov space (block GMRES) and one batched FMM tree per matvec
  (`lfmm3d` with `nd = K`).

This decouples the dominant GMRES cost from being paid $K$ times down to roughly once per
group.

## Scope (this deliverable)

**Steps 1–6 only: grouping → shared mesh → batched RHS → vectorized matvec → block GMRES,
stopping at the layer densities $\Sigma$.** The post-refinement evaluation and final
contraction to $V_{ijkl}$ (Step 7, Sec. 3 / Algorithm 1 steps 4–5 of the paper) are a
follow-on, not part of this branch's first deliverable.

## Governing constraint

Block GMRES on a single operator requires **every RHS in the block to share the same
discretization** — identical panelization, identical $N$ — so that $A$ is literally one
matrix. This is why grouping is by center: co-located sources admit one shared mesh.
Different centers refine different faces → different meshes → separate blocks. **The grouping
is the blocking.**

## Current code (verified)

- `lhs_dielectric_box3d_fmm3d_corrected(interface, fmm_tol, up_tol, max_order; …)`
  (`src/solver/dielectric_box3d.jl:48`) builds $A$ as a `LinearMap{Float64}` from geometry +
  contrast only. Its matvec is `D_base * charges + corrections * charges` plus the diagonal
  $\tfrac12(\epsilon_o+\epsilon_i)/(\epsilon_o-\epsilon_i)$ term.
- `D_base` (`laplace3d_DT_fmm3d`, `src/kernel/laplace3d.jl:84`) wraps a single
  `lfmm3d(thresh, sources, charges = weights .* charges, pg = 2)` call. **FMM3D 1.0.1
  accepts a charge *matrix* `(N, K)` via `nd = K`, building the tree once.**
- `corrections` (`laplace3d_DT_fmm3d_corrected`, `src/kernel/laplace3d_near.jl:376`) is a
  **sparse matrix** — `corrections * X` is already a batched mat-mat.
- RHS-driven adaptive refinement is driven by a scalar Neumann-data function
  `rhs(p, n)` in `multi_dielectric_box3d_rhs_adaptive` (`src/shape/box3d_multi.jl:388`),
  refining each panel until its tensor-product interpolation error of `rhs` is below
  `rhs_atol`.
- Per-source RHS assembly: `rhs_dielectric_box3d*` (`src/solver/dielectric_box3d.jl:95+`)
  returns a length-$N$ vector.
- `Krylov.block_gmres` is available (Krylov 0.10.6).

## Architecture

Six units, each independently testable.

### Unit 1 — `RHSGroup` (group assembly)
A container for one center's batch: the shared center $i$, the $K$ neighbor densities
$\{\rho_{ij}\}$ (as `VolumeSource`s or `PointSource`s), and bookkeeping (which $(i,j)$ each
column is). Construction takes orbital centers + a neighbor list and emits one `RHSGroup`
per center.
- **Input:** orbital data + neighbor list.
- **Output:** `RHSGroup` with `Vector{<:AbstractSource}` of length $K$.
- **Depends on:** existing source types only.

### Unit 2 — Shared per-group panelization (union/envelope refinement)
Extend the RHS-driven refinement so a panel is refined until the interpolation-error
criterion holds for **all** group members simultaneously — i.e. the per-panel error driver
is the max over members, $\max_{j} \|f_{ij} - g_P[f_{ij}]\|_{L^2(P)} < \texttt{rhs\_atol}$.
Realized by passing the group's RHS as a vector-valued driver to a new
`multi_dielectric_box3d_rhs_adaptive` method that accepts `Vector{<:AbstractSource}` (or a
vector-valued `rhs` function) and takes the worst member per panel.
- **Input:** boxes/epses + the group's sources + `rhs_atol`.
- **Output:** one `DielectricInterface` resolving every member.
- **Depends on:** `box3d_multi.jl` refinement, screening.

### Unit 3 — $A$ once per group
Direct reuse of `lhs_dielectric_box3d_fmm3d_corrected` on the Unit-2 interface. The neighbor
list, sparse `corrections`, and FMM source setup are built once and reused across all $K$
RHS. No change needed beyond passing the shared interface.

### Unit 4 — Batched RHS assembly $F$
New methods `rhs_dielectric_box3d*(interface, sources::Vector{<:AbstractSource}, …)` returning
an $N \times K$ matrix whose column $k$ is $f_{i,j_k}$. The TKM/FMM evaluation of
$-\partial_{\V n} u_{\mathrm{inc}}$ batches over members (same target nodes, same kernel;
`lfmm3d` with `nd = K`).
- **Input:** shared interface + group sources.
- **Output:** `Matrix{Float64}` of size $N \times K$.

### Unit 5 — Matrix-capable vectorized matvec ⭐
The crux. Introduce an operator whose `mul!(Y::Matrix, op, X::Matrix)`:
1. one batched `lfmm3d(thresh, sources, charges = weights .* X, pg = 2)` with `nd = K`,
   then contract the gradient with normals per column;
2. `corrections * X` (sparse mat-mat);
3. add the diagonal $\Gamma^{-1}$ term column-wise.

This must be a type that `block_gmres` drives with a matrix RHS *without* falling back to a
per-column FMM loop (that fallback would forfeit the FMM speedup — the whole point). Concretely
a small struct implementing `Base.size` and `LinearAlgebra.mul!(::Matrix, ::Op, ::Matrix)`
(and the vector overloads, so the single-RHS path is unchanged). The existing per-vector
`LinearMap` is retained for $K=1$ / backward compatibility.
- **Input:** same data as `lhs_dielectric_box3d_fmm3d_corrected`.
- **Output:** an operator usable by both `gmres` and `block_gmres`.

### Unit 6 — Block GMRES solve
`block_gmres(op, F; …)` → $\Sigma = [\sigma_{i1}\,\cdots\,\sigma_{iK}]$, exposing tolerance,
restart, and max-iteration parameters consistent with the current `gmres` call. A thin driver
`solve_dielectric_box3d_group(group; …)` wires Units 2–6 together and returns $\Sigma$ plus the
shared interface (needed by the eventual Step-7 evaluation).

## Data flow

```
orbital centers + neighbor list
        │  Unit 1
        ▼
   RHSGroup {center i, sources[1..K]}
        │  Unit 2  (union/envelope refinement)
        ▼
   DielectricInterface (shared)
        ├── Unit 3 ──► A  (LHS operator, built once)
        └── Unit 4 ──► F  (N × K RHS matrix)
                        │  Unit 5 (vectorized matvec) + Unit 6 (block GMRES)
                        ▼
                   Σ = A \ F   (N × K layer densities)
```

## Error handling / edge cases

- **$K = 1$** must route through (or exactly reproduce) the existing single-RHS path,
  bit-for-bit, so nothing regresses.
- **Empty group** ($K = 0$, no neighbors): return an empty $\Sigma$ without calling the solver.
- **FMM `nd` shape conventions**: verify the exact `lfmm3d` matrix layout (`(nd, N)` vs
  `(N, nd)` and gradient output shape) against FMM3D 1.0.1 before wiring Unit 5; an
  off-by-transpose here is silent and corrupts results.
- **Block GMRES non-convergence / disparate per-column scaling**: members with very different
  norms can stress the shared Krylov space; expose tolerance/restart and surface the per-column
  residuals so non-convergence is visible.

## Testing strategy

1. **Single-RHS regression:** $K=1$ block path reproduces the current single solve to machine
   precision on an existing box3d test case.
2. **Batched-matvec correctness:** Unit-5 `mul!(Y, op, X)` equals column-wise application of
   the existing per-vector matvec (to ~1e-12).
3. **Block vs looped solve:** `block_gmres(op, F)` matches $K$ independent `gmres(op, f_k)`
   solves to GMRES tolerance.
4. **Shared-mesh adequacy:** the union-refined mesh drives every member's RHS interpolation
   error below `rhs_atol` (reuse the $\mathcal E_f$ diagnostic, Eq. 28 of the paper).
5. **Performance smoke test:** confirm one batched FMM per block iteration (not $K$), e.g. via
   FMM call counting or wall-clock scaling vs $K$.

## Out of scope (follow-on)

- Step 7: post-refinement evaluation + contraction to $V_{ijkl}$.
- Cross-center batching / global low-rank RHS compression.
- Choosing/forming the neighbor list from real orbital data (assumed provided as input here).
