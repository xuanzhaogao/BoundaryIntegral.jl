# Near-Correction Improvements: Source-Side Reuse + Adaptive Edge-Edge Correction

Date: 2026-05-18 (v2 revision after Codex headless review)
Status: Approved (brainstorming) — pending implementation plan.

## 1. Motivation

The 3D Laplace near-singular quadrature correction in
`src/kernel/laplace3d_near_upsampling.jl` currently has two limitations:

1. **Per-pair recomputation of source-panel data.** The outer loop iterates
   over `(src_panel i, trg_panel j)` pairs. For a single source panel
   with `k` neighboring target panels, the upsampled-grid positions,
   scale, and the application of the interpolation matrix `Ex` are
   recomputed `k` times. The pair `(ns_up, ws_up, Ex)` is already cached
   globally per `n_up` value in `_build_upsampling_cache`
   (`laplace3d_near_upsampling.jl:143`), but the materialization of the
   per-source "moments-to-nodal" tensor is per pair.

2. **Edge-edge touching pairs are skipped.** `build_neighbor_list`
   (`laplace3d_near.jl:269`) filters out any pair where either panel has
   `is_edge == true`. For two panels that share an edge or corner —
   typically a cross-face pair on a box — the kernel is nearly
   singular at the shared boundary and the smooth Gauss-Legendre rule
   used by FMM is inaccurate there. The current code has no
   replacement; that contribution stays uncorrected.

This document specifies a two-tier near correction that (a) refactors
the upsampling path into a per-source moments-style form (clarity,
allocation reduction, and a wall-time win from BLAS-3 batching), and
(b) adds an opt-in adaptive subdivision path for touching pairs.

## 2. Background

`docs/near_panels.md` describes the kernel-weighted polynomial moments
framework used in `fmm3dbie` (Greengard, O'Neil, Rachh, Vico 2021):

  Iʲ_{nm}(x) = ∫_{T₀} K(x, Xʲ(u,v)) · K_{nm}(u,v) · Jʲ(u,v) du dv,
  aʲ_ℓ(x)   = Σ_{nm} Iʲ_{nm}(x) · V_{(nm),ℓ}.

For flat tensor-product square panels with a Lagrange basis at the
panel's own GL nodes, `V` is the identity *for that panel*, and the
moments collapse to the upsampling formula already used here. The two
tasks expressed in this language are:

- **Task 1** — restructure so that, for each source panel `j`, the
  geometry, upsampled nodes, and the "ws · Ex⊗Ex · scale" tensor are
  computed once and reused across every near target on every
  neighboring target panel.

- **Task 2** — for touching pairs (where smooth-quadrature moments are
  inaccurate), evaluate the moments by adaptive quadtree subdivision on
  `[-1,1]²` instead.

Caveat: `V = I` per panel does **not** imply the upsampling cache may
be built once globally. Different source panels may use different
`gl_xs / gl_ws` (varquad — exercised by
`test/kernel/laplace3d_near.jl:258`). The current global cache assumes
uniform GL nodes across the interface. The per-source `SourceCache`
introduced below fixes this.

## 3. Scope and Non-Goals

In scope:

- 3D Laplace, flat square panels (`FlatPanel{T,3}`).
- DT (target-normal-derivative) and D (source-normal-derivative) layer
  operators — the two modes already handled by
  `laplace3d_panel_upsampled`.
- Edge-edge correction strictly for **touching** pairs (`i ≠ j` sharing
  at least one corner within geometric tolerance).
- Refactor of `_laplace3d_corrections` and `build_neighbor_list`.

Out of scope (deferred to follow-ups):

- Self-interaction `i == j` (DT vanishes for flat same-panel pairs).
- Singularity-aware initial subdivision (v1 adaptive is pure
  error-driven).
- 2D Laplace and Helmholtz kernels.
- Curved patches.
- Replacing the upsampling path entirely (design keeps both tiers).
- Off-surface near evaluation in `laplace3d_near_hcubature.jl` (uses
  `_refine_interface_for_targets`, a separate path — see §7).
- `T <: AbstractFloat` other than `Float64` for the public corrected
  wrappers (`laplace3d_DT/D_fmm3d_corrected` are `Float64`-only by FMM3D
  constraint, `laplace3d_near.jl:331,350`).

## 4. Architecture

```
build_neighbor_list(..., correct_edges)
        │
        ▼
(upsample_dict, adaptive_dict)                  # two dicts; never overlap
        │
        ▼
_laplace3d_corrections(interface, upsample_dict, adaptive_dict, mode)
   for source panel i with ≥1 neighbor:
     if i appears in upsample_dict:
        build SourceCache_i (per-source p_up, Mt, Ex from panel.gl_xs)
     for each (i, j) in upsample_dict for this i:
        Kmat[α, t] = K(t, p_up[α])     # one batched kernel-eval pass
        K_block    = Mt * Kmat         # BLAS-3, shape (n_quad², n_p_trg)
        K_direct   = laplace3d_{DT,D}_panel(panel_i, panel_j)
        push (transpose(K_block) - K_direct) into per-thread triplets
     for each (i, j) in adaptive_dict for this i:
        for t in panel_j.points:
            K_block[t,:] = adaptive_panel_moments_inplace!(...)
        K_direct = laplace3d_{DT,D}_panel(panel_i, panel_j)
        push (K_block - K_direct) into per-thread triplets
   merge → SparseMatrixCSC
```

Key invariants:

- Both paths produce a block of shape `(num_points(panel_j), n_quad²)`
  before the `K_direct` subtraction.
- The sparse output's structural sparsity and value at every entry must
  agree with the current implementation on the union of
  `upsample_dict ∪ "previously skipped edge pairs treated as zero"`,
  i.e. the `correct_edges = false` path is value-equivalent to today
  up to summation roundoff.
- Parallelism is over source panels. Per-source threading is the
  default but is benchmarked against per-pair threading on a
  representative geometry (§11) before being declared the winner.
- `correct_edges = false` (the default) emits an empty
  `adaptive_dict`, leaves `is_edge` filtering exactly as today, and
  produces a sparse correction matrix matching the current code to
  roundoff (§8 test 1).

## 5. Components

### 5.1 Neighbor-list output: two dicts (revised)

`build_neighbor_list` returns a `NamedTuple`:

```julia
(; upsample = Dict{Tuple{Int,Int}, Int},                  # value: n_up
   adaptive = Dict{Tuple{Int,Int}, AdaptiveConfig})
```

where

```julia
struct AdaptiveConfig
    atol::Float64
    rtol::Float64
    n_GL::Int
    max_depth::Int
end
```

Rationale (from review): a tagged-struct `NearMethod(kind::Symbol, …)`
is type-unstable in hot loops and admits invalid states. Two dicts
also drop in cleanly because `_build_upsampling_cache` already consumes
`Dict{Tuple{Int,Int}, Int}` (`laplace3d_near_upsampling.jl:143`); only
the variable name changes.

A pair appears in exactly one of the two dicts; an assertion in
`_laplace3d_corrections` verifies disjointness.

### 5.2 `build_neighbor_list` (modified, two-phase candidate discovery)

New kwarg `correct_edges::Bool = false`. Two-phase candidate
discovery is required because the current single-phase KDTree scan is
over **GL quadrature nodes** (interior points,
`laplace3d_near.jl:265`), and two panels that share only an edge or
corner can have *no* GL nodes within `r_i`. The corner test would
then never fire.

Phase A (current logic, preserved):
- Build the GL-node KDTree.
- For each source panel `i`, query nodes within
  `r_i = range_factor · l_i / n_quad_i`.
- Group by target panel `j`; this populates the candidate set for
  upsampling.

Phase B (new, only if `correct_edges == true`):
- Build a second KDTree over **panel corners** (4 per panel).
- For each source panel `i`, query corners within
  `tol_corner = √eps(T) · max_panel_length(interface)`. (A loose bound;
  for box-corner geometries shared corners are exact-equal in float, so
  any positive tolerance works.)
- Promote any candidate `(i, j)` discovered here to the `adaptive`
  dict.

Pair classification, applied in order:

1. `i == j` → skip.
2. `coplanar same-plane` skip — **preserve the existing predicate
   verbatim** from `laplace3d_near.jl:291–296`: normals dot > `1 − √eps`
   AND `|plane_offsets[i] − plane_offsets[j]| ≤ √eps · max(1, l_i)`.
   This predicate is currently applied uniformly to both modes (the
   wrappers in `laplace3d_near.jl:331,350` use the same builder for DT
   and D). We keep this; a mode-aware variant is an explicit
   non-goal (§3) and tracked as O3.
3. If `correct_edges == false`:
   - either panel has `is_edge == true` → skip;
   - otherwise → upsample dict with `n_up` from existing
     `check_quad_order3d` logic.
4. If `correct_edges == true`:
   - `is_edge` filter dropped;
   - Phase B promoted this pair → adaptive dict with the
     `AdaptiveConfig` passed from the public API;
   - otherwise → upsample dict (same `n_up` selection as today).

Note on existing defaults: `dielectric_box3d.jl:53` defaults
`include_edges_src/trg = true`, but `laplace3d_near.jl:336,355` defaults
them to `false`. The new `correct_edges` flag is orthogonal to
`include_edges_*` and we keep both for backward compatibility; their
interaction is spelled out in §6.

### 5.3 `SourceCache{T}` (new, per-source)

In `src/kernel/laplace3d_near_upsampling.jl`:

```julia
struct SourceCache{T}
    panel::FlatPanel{T,3}
    n_up::Int
    p_up::Vector{NTuple{3,T}}   # length n_up²; upsampled physical positions
    Mt::Matrix{T}               # n_quad² × n_up²
end

build_source_cache(panel, n_up) :: SourceCache
```

Construction:

- `ns_up, ws_up = gausslegendre(n_up)`.
- `Ex = interp_matrix_1d_gl(panel.gl_xs, panel.gl_ws, ns_up)` — built
  from *this panel's own GL nodes*, fixing the varquad issue.
- `Mt[m, α]` where `α ↔ (i_up, j_up)` and `m ↔ (m_x, m_y)`, both
  flattened with the **outer = first index, inner = second** convention
  matching `rect_panel3d_discretize` (`src/shape/box3d.jl:47`) and the
  existing `K_up[ti, idx]` ordering in
  `laplace3d_near_upsampling.jl:127`:

    α  = (i_up - 1) * n_up + j_up
    m  = (m_x  - 1) * n_quad + m_y
    Mt[m, α] = scale · ws_up[i_up] · ws_up[j_up]
              · Ex[i_up, m_x] · Ex[j_up, m_y]

  This is exactly equivalent to `transpose(Ex) * Dw * Ex` flattened
  with the same convention; a regression test (§8.6) pins this.
- `p_up[α] = cc + bma · (ns_up[i_up]/2) + dma · (ns_up[j_up]/2)`.

Per-target evaluation, batched over a target panel `j`:

    Kmat[α, t] = K(panel_j.points[t], p_up[α])  for α=1..n_up², t=1..n_p_trg
    K_block_T  = Mt * Kmat                       # (n_quad², n_p_trg) = BLAS-3
    K_block    = transpose(K_block_T)            # (n_p_trg, n_quad²)

The kernel-eval phase is the dominant cost and is itself BLAS-friendly
when fused with the matmul if a future optimization warrants. v1 keeps
them separate.

### 5.4 `adaptive_panel_moments_inplace!` (new)

New file `src/kernel/laplace3d_near_adaptive.jl`:

```julia
function adaptive_panel_moments_inplace!(
    K_row::AbstractVector{T},         # length n_quad², output, zeroed by caller
    panel_src::FlatPanel{T,3},
    point_trg::NTuple{3,T},
    trg_normal::NTuple{3,T},          # used for :DT; ignored for :D
    mode::Symbol,                     # :DT or :D
    cfg::AdaptiveConfig,
) where T
```

Recursive quadtree on `(u,v) ∈ [-1,1]²`:

1. Compute parent moments via `n_GL × n_GL` GL on the current cell
   `[u_lo, u_hi] × [v_lo, v_hi]`.
2. Compute children moments on each of 4 sub-cells with the same rule.
3. `err = maximum(abs.(parent .- sum(children)))` across all `n_quad²`
   moments.
4. `tol_local = max(atol, rtol · maximum(abs.(parent)))`.
5. If `err < tol_local` or `depth == max_depth`: accumulate the
   children sum into `K_row` and return.
6. Else recurse on each child with the same `cfg` (atol and rtol are
   **not** split per level; the local-error sum across all leaves is
   bounded by O(n_leaves · tol_local), which is acceptable in practice
   — see O1 below for the alternative).

At each GL node `(u, v)` on a leaf cell:

- physical `y = X^j(u, v)`,
- kernel `K(point_trg, y)` (mode-dispatched: DT uses `trg_normal`, D
  uses `panel_src.normal`),
- Lagrange basis values `L_{m_x}(u)`, `L_{m_y}(v)` via
  `barycentric_row!` against `panel_src.gl_xs` and
  `panel_src.bary_weights`,
- `scale_leaf = ((u_hi - u_lo)/2) · ((v_hi - v_lo)/2) · scale_panel`.

Output `K_row[m]` accumulates `weight · K · L_{m_x}(u) · L_{m_y}(v) ·
scale_leaf` over all leaf GL nodes.

`max_depth` behavior: the routine returns the deepest-leaf children
sum (never the parent). If `max_depth` is hit, it records the pair in
a *thread-local* `Set{Tuple{Int,Int}}` and reports it back to the
caller; the caller merges thread-local sets and emits exactly one
`@warn` per pair after the parallel loop.

### 5.5 `_laplace3d_corrections` (restructured)

Outer loop over source panel indices `i ∈ keys(upsample_dict) ∪
keys(adaptive_dict)`. For each `i`:

- If `i` has any upsample neighbor: build `SourceCache_i` with the
  largest `n_up` requested by any of its upsample neighbors.
- For each `(i, j)` in `upsample_dict`: batch all target points of
  panel `j` into `Kmat`, compute `K_block` by BLAS-3, subtract
  `K_direct`, push triplets.
- For each `(i, j)` in `adaptive_dict`: per-target call to
  `adaptive_panel_moments_inplace!`, accumulate to `K_block`,
  subtract `K_direct`, push triplets.

Threading: `@threads :dynamic` over source panels by default (work per
source varies with number of neighbors and adaptive load). If
benchmarks (§11) show per-pair `@threads :static` wins on
representative geometries, swap. The choice is configurable via an
internal env-var `BI_NEAR_SCHEDULING ∈ {:source, :pair}` for testing,
but the **default** is decided by the benchmark before merge.

Per-thread scratch:

- `Kmat_tl`     :: `Matrix{T}(n_up_max², n_p_trg_max)`
- `Kblock_tl`   :: `Matrix{T}(n_quad_max², n_p_trg_max)` (and
  transpose buffer)
- `rows_tl, cols_tl, vals_tl` :: sparse-triplet vectors
- `Krow_tl`     :: `Vector{T}(n_quad_max²)` for the adaptive path
- `warn_set_tl` :: `Set{Tuple{Int,Int}}` for max-depth records

### 5.6 Public API

`laplace3d_DT_fmm3d_corrected` and `laplace3d_D_fmm3d_corrected`
(`laplace3d_near.jl:331,350`) gain kwargs (Float64 only):

```julia
correct_edges::Bool       = false
adaptive_atol::Float64    = up_tol
adaptive_rtol::Float64    = sqrt(eps(Float64))
adaptive_n_GL::Int        = 0           # 0 → use source panel's n_quad
adaptive_max_depth::Int   = 20
```

`adaptive_*` kwargs are consulted only when `correct_edges = true`.

`lhs_dielectric_box3d_fmm3d_corrected`
(`src/solver/dielectric_box3d.jl:48`) forwards the same kwargs and
preserves its `include_edges_src/trg = true` default.

Signatures stay backward-compatible: omitting all new kwargs
reproduces current behavior modulo summation order (§8 test 1).

## 6. Data Flow

Single source panel `i`, assembly time:

1. `ups_neighbors = [(i,j) in upsample_dict]`,
   `adp_neighbors = [(i,j) in adaptive_dict]`.
2. If non-empty `ups_neighbors`: build `SourceCache_i` with
   `n_up = max(values for these neighbors)`.
3. For each `(i, j) ∈ ups_neighbors`:
   - Build `Kmat[α, t]` over all target points `t ∈ panel_j.points`.
   - `K_block_T = Mt * Kmat`, then transpose into `K_block[t, :]`.
   - Compute `K_direct = laplace3d_{DT,D}_panel(panel_i, panel_j)`
     (existing routine).
   - Push `(K_block − K_direct)` triplets.
4. For each `(i, j) ∈ adp_neighbors`:
   - For each target `t`: zero `K_block[t, :]`, call
     `adaptive_panel_moments_inplace!`.
   - Compute `K_direct`.
   - Push `(K_block − K_direct)` triplets.

Matvec time: unchanged.

```
u = laplace3d_{DT,D}_fmm3d(...) · σ  +  corrections_sparse · σ
```

## 7. Error Handling, Edge Cases, and Interactions

- **Adaptive non-convergence at `max_depth`.** Thread-local
  `Set{Tuple{Int,Int}}` records each affected pair; merged after the
  parallel loop; emit exactly one `@warn` per affected pair. (Codex
  flagged that per-thread emission can duplicate; the merge-then-emit
  pattern fixes it.)
- **Empty dicts.** `correct_edges = true` with no touching pairs in
  the geometry — legal; `adaptive_dict` is empty; identical behavior
  to `false` plus the `is_edge` filter being dropped.
- **Variable `n_quad` across panels.** Handled by building `Ex` from
  each source panel's own `gl_xs/gl_ws` inside `SourceCache`. The
  global `_build_upsampling_cache` is removed.
- **`is_edge` ↔ `correct_edges` interaction.** The two flags are
  orthogonal. With `correct_edges = false`, the existing
  `include_edges_src/trg` filters behave exactly as today. With
  `correct_edges = true`, `is_edge` is ignored in
  `build_neighbor_list`; the touching-pair test classifies pairs by
  geometry, which is the correct notion.
- **Interaction with `laplace3d_near_hcubature.jl`.** That file
  handles **off-surface** near evaluation (target ∉ Γ) via
  `_refine_interface_for_targets` and is independent of the corrected
  on-surface operator. v1 leaves it untouched. A follow-up may unify
  the two adaptive paths.
- **`Float64`-only public API.** FMM3D library is `Float64`. The
  `T`-generic routines (`SourceCache{T}`,
  `adaptive_panel_moments_inplace!`) remain generic and can be used
  in tests with `Float32` for stress testing, but the corrected
  wrappers stay `Float64`-only.

## 8. Testing

1. **Refactor regression.** With `correct_edges = false` on the
   existing test cases (uniform `n_quad`), the sparse correction
   matrix must agree with the current code to roundoff (Frobenius-norm
   relative diff < `1e-12` in `Float64`). Direct equality against a
   frozen reference matrix on a small dielectric box.

2. **Varquad correctness.** Reuse the existing
   `test/kernel/laplace3d_near.jl:258` varquad fixture. Build the
   correction matrix with the new code; compare against a direct
   per-panel `hcubature` evaluation. The current global cache code
   path silently used `interface.panels[1].gl_xs` for *all* panels;
   confirm the new per-source `Ex` makes varquad correct and matches
   direct evaluation.

3. **`SourceCache` reuse counter.** Wrap `build_source_cache` in a
   debug-mode counter; assert it is called exactly once per source
   panel that has at least one upsample neighbor.

4. **Adaptive moments unit test.** Fixed source panel, single
   off-panel target. Compute each of the `n_quad²` moments with
   `adaptive_panel_moments_inplace!` and with high-order `hcubature`
   from `HCubature.jl`. Assert agreement to
   `max(atol, rtol · |reference|)`.

5. **Corner-pair discovery test.** Construct a 2-panel cross-face
   geometry where the two panels share an edge but no GL nodes are
   within `r_i` of each other. Assert that `build_neighbor_list(...,
   correct_edges = true)` returns the pair in `adaptive_dict` and
   that the old single-phase code path would have returned no pair.

6. **`Mt` indexing regression.** On a single panel, construct `Mt`
   per the formula in §5.3, build `Dw` arbitrarily, and verify that
   `Mt[:, :] * vec(Dw)` (with the documented α-flatten) equals
   `vec(transpose(Ex) * Dw * Ex)` (with the documented m-flatten),
   pinning the indexing convention.

7. **Cross-face touching regression (headline).** 2-panel box-corner
   geometry. Compare DT applied to a smooth density via:
   (a) brute-force `hcubature` over both panels (reference),
   (b) new corrections with `correct_edges = false`,
   (c) new corrections with `correct_edges = true`.
   Assert (c) is at least 4 orders of magnitude closer to (a) than
   (b) at `adaptive_atol = 1e-8`.

8. **End-to-end dielectric box.** Rerun the dielectric-box example
   with `correct_edges = true`. Compare GMRES residual at convergence
   and/or solution error against a known analytic case (point-charge
   inside a dielectric box) to the `correct_edges = false` baseline.
   Acceptance: measurable solution-error reduction.

## 9. Open Questions

- **O1. Atol behavior across recursion levels.** §5.4 specifies a
  flat per-cell threshold (`tol_local = max(atol, rtol · max(|parent|))`)
  without splitting across depth. Classic h-adaptive quadrature uses
  `atol/2^depth` or sum-of-leaves bookkeeping. The flat rule is
  cheaper and matches `HCubature.jl`'s defaults; revisit if the
  cross-face regression (§8.7) fails to converge.

- **O2. Per-source vs per-pair threading.** Benchmark on a 6-face box
  with refined corners (n_panels ~10²–10³) before merging. Default
  to whichever wins. Both schedules retained behind
  `BI_NEAR_SCHEDULING`.

- **O3. Mode-aware DT/D-vanishing predicate.** §5.2 preserves the
  existing predicate verbatim for both modes. A mode-aware predicate
  (`:D` depends on `n_src · (x − y)`) would skip slightly more pairs
  for D. Out of scope for v1; revisit if D-mode benchmarks reveal
  wasted work.

## 10. Files Touched

- `src/kernel/laplace3d_near.jl` — `build_neighbor_list` signature
  (returns NamedTuple of two dicts), two-phase candidate discovery
  (panel-corner KDTree), thread new kwargs through
  `laplace3d_DT_fmm3d_corrected` / `laplace3d_D_fmm3d_corrected`.
- `src/kernel/laplace3d_near_upsampling.jl` — introduce `SourceCache`
  / `build_source_cache`; restructure `_laplace3d_corrections` for
  per-source loop with BLAS-3 batching over target panel; remove
  global `_build_upsampling_cache`.
- `src/kernel/laplace3d_near_adaptive.jl` — new file with
  `AdaptiveConfig`, `adaptive_panel_moments_inplace!`.
- `src/BoundaryIntegral.jl` — `include` the new file.
- `src/solver/dielectric_box3d.jl` — forward the new kwargs from
  `lhs_dielectric_box3d_fmm3d_corrected`.
- `test/` — new test files for §8 items 2, 3, 4, 5, 6, 7; extend
  the dielectric-box example test for item 8.
- `src/kernel/laplace3d_near_hcubature.jl` — untouched in v1 (§7);
  document the non-interaction in its header comment.

## 11. Acceptance Criteria

- All existing tests pass with `correct_edges = false` (default).
- New tests (items 1–8 in §8) pass.
- Benchmark for O2 (per-source vs per-pair threading) is committed
  to `test/benchmarks/` and reports a chosen winner; the default
  scheduling matches the winner.
- Wall-time for assembly with `correct_edges = false` on a
  representative dielectric-box case (~10³ panels) is no slower than
  the current implementation. The BLAS-3 + per-source design is
  expected to give a meaningful speedup; we will not gate merge on a
  specific number, but a regression here requires investigation.
- The dielectric-box example with `correct_edges = true` shows
  measurable solution-error reduction over the `false` baseline.
