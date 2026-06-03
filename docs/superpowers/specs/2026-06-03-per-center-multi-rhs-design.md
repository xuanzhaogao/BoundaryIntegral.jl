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

**Step 0 + Steps 1–6: parse system input → grouping → shared mesh → batched RHS →
vectorized matvec → block GMRES, stopping at the layer densities $\Sigma$.** The
post-refinement evaluation and final contraction to $V_{ijkl}$ (Step 7, Sec. 3 / Algorithm 1
steps 4–5 of the paper) are a follow-on, not part of this branch's first deliverable.

## Step 0 — System input file

A single human-editable text file is the entry point. It describes (a) the dielectric
geometry, (b) the orbitals — each a path to an `.xsf` holding one orbital $\varphi_i$ on a
grid — and (c) the grouping rule that produces the per-center batches. Pair densities
$\rho_{ij} = \varphi_i\varphi_j$ are **formed in-code** by pointwise product; the file lists
orbitals, not products.

### Format

`.xsf`-style keyword blocks (`BEGIN_<NAME>` … `END_<NAME>`), `#` comments, blank lines
ignored. Proposed extension `.bie`. Paths are resolved relative to the input file's directory.

```
# BoundaryIntegral system input
UNITS bohr                      # bohr | angstrom — applies to all lengths/positions below

BEGIN_DIELECTRICS
EPS_OUT 1.0                     # permittivity outside all boxes (vacuum)
# one row per box:  cx cy cz    Lx Ly Lz    eps
  0.0 0.0 0.0     20.0 20.0 4.0    11.7
  0.0 0.0 8.0     20.0 20.0 4.0     3.9
END_DIELECTRICS

BEGIN_ORBITALS
# one row per orbital:  id   xsf_path   [atom_index]
# center is read from the .xsf PRIMCOORD; atom_index selects which atom and is
# required only when PRIMCOORD lists more than one atom.
  1   wann_00001.xsf
  2   wann_00002.xsf   5
END_ORBITALS

BEGIN_GROUPING
CUTOFF 8.0                      # default: center i groups with every j (including i)
                                # whose center lies within 8.0 of i's center
# optional explicit overrides — one per line:  i : j1 j2 j3 ...
# OVERRIDE replaces the cutoff-derived neighbor set for center i
  3 : 3 4 7
END_GROUPING
```

### Parser output

A struct (e.g. `SystemInput`) carrying:
- `unit_scale` (factor to internal units),
- `boxes::Vector{NamedTuple}` (`center, Lx, Ly, Lz`), `epses::Vector`, `eps_out` — feeding
  the existing `multi_dielectric_box3d_*` machinery directly,
- `orbitals`: `id ↦ (xsf_path, center)` with `center` taken from the `.xsf` `PRIMCOORD`,
- `groups`: for each center $i$, the resolved neighbor list $\{j\}$ (cutoff, then overrides).

This is exactly the input Unit 1 consumes: each group $\{(i,j)\}$ becomes one `RHSGroup`,
with $\rho_{ij}$ produced by loading $\varphi_i, \varphi_j$ from their `.xsf` grids and
multiplying.

### Assumptions & edge cases

- **Common product grid.** Forming $\varphi_i\varphi_j$ requires both orbitals on the same
  grid. Wannier90 typically writes every WF on one supercell grid (identical `PRIMVEC`,
  dimensions, origin), making the product a pointwise multiply. The loader **validates grid
  compatibility** for each pair; if grids differ, the fallback is to resample $\varphi_j$ onto
  $\varphi_i$'s grid via the existing trilinear interpolation
  (`_datagrid_trilinear_value`). The product $\rho_{ij}$ is supported only where both factors
  are non-negligible, so it is truncated/thresholded to that overlap region (reusing
  `VolumeSource`'s `tol`).
- **Center from PRIMCOORD.** If a `.xsf` PRIMCOORD lists exactly one atom, `atom_index` may be
  omitted. If it lists several and `atom_index` is absent, that is a parse error (no silent
  guess). *(Open question flagged for review: whether to allow a density-centroid fallback
  here instead of erroring.)*
- **Self pairs.** The cutoff set includes $i$ itself, so the self-density $\rho_{ii}$ is part
  of each group.
- **Units.** `UNITS` scales box geometry and the cutoff consistently with the `.xsf`
  coordinate units; mismatched `.xsf` units are out of scope (assumed consistent with `UNITS`).

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

### Unit 0 — System input parser
Parse the Step 0 `.bie` file into a `SystemInput` (see Step 0 above): unit scale, dielectric
boxes + permittivities, orbital table (`id ↦ xsf_path, center`), and the resolved per-center
neighbor groups. Centers are read from each `.xsf` PRIMCOORD.
- **Input:** path to `.bie` file.
- **Output:** `SystemInput`.
- **Depends on:** `read_xsf` (`src/utils/xsf_reader.jl`) for centers/grids.

### Unit 1 — `RHSGroup` (group assembly)
A container for one center's batch: the shared center $i$, the $K$ neighbor densities
$\{\rho_{ij}\}$ (as `VolumeSource`s), and bookkeeping (which $(i,j)$ each column is).
Construction consumes a `SystemInput` group: for each $j$ in $\mathrm{neigh}(i)$, load
$\varphi_i,\varphi_j$ from their `.xsf` grids and form $\rho_{ij}=\varphi_i\varphi_j$ as a
`VolumeSource` (validating/aligning grids per the Step-0 assumptions).
- **Input:** `SystemInput` + a center id.
- **Output:** `RHSGroup` with `Vector{VolumeSource}` of length $K$.
- **Depends on:** Unit 0, source types, `_datagrid_trilinear_value` (fallback resampling).

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
   system .bie file
        │  Unit 0
        ▼
   SystemInput {boxes, epses, eps_out, orbitals, groups}
        │  Unit 1   (load .xsf, form rho_ij = phi_i * phi_j)
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

0. **Parser round-trip:** a fixture `.bie` file parses to the expected `SystemInput`
   (boxes, epses, eps_out, orbital table, resolved groups incl. cutoff + override); malformed
   inputs (missing block, multi-atom PRIMCOORD without `atom_index`) raise clear errors.
   Pair-density formation $\rho_{ij}=\varphi_i\varphi_j$ matches a pointwise product on a
   shared-grid fixture, and the resample fallback matches trilinear interpolation on a
   mismatched-grid fixture.
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
