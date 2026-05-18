# Near-Correction Improvements: Source-Side Reuse + Adaptive Edge-Edge Correction

Date: 2026-05-18
Status: Approved (brainstorming) — pending implementation plan.

## 1. Motivation

The 3D Laplace near-singular quadrature correction in
`src/kernel/laplace3d_near_upsampling.jl` currently has two limitations:

1. **Per-pair recomputation of source-panel data.** The outer loop iterates
   over `(src_panel i, trg_panel j)` pairs. For a single source panel with
   `k` neighboring target panels, the upsampled-grid positions, scale, and
   per-source geometry are recomputed `k` times. The interpolation matrix
   `Ex` and the upsampled GL nodes are already cached globally per `n_up`
   value, but the application of that cache (and the materialization of the
   "moments-to-nodal" transformation) is per pair.

2. **Edge-edge touching pairs are skipped.** `build_neighbor_list` filters
   out any pair where either panel is flagged `is_edge`. For two panels
   that share an edge or corner — typically a cross-face pair on a box —
   the kernel is nearly singular at the shared boundary and the smooth
   Gauss-Legendre rule used by FMM is inaccurate there. The current code
   has no replacement; that contribution stays uncorrected.

This document specifies a two-tier near correction that (a) refactors the
upsampling path into a per-source moments-style form (yielding clarity,
allocation reduction, and a modest wall-time win), and (b) adds an
opt-in adaptive subdivision path for touching pairs.

## 2. Background

`docs/near_panels.md` describes the kernel-weighted polynomial moments
framework used in `fmm3dbie` (Greengard, O'Neil, Rachh, Vico 2021):

  Iʲ_{nm}(x) = ∫_{T₀} K(x, Xʲ(u,v)) · K_{nm}(u,v) · Jʲ(u,v) du dv,
  aʲ_ℓ(x)   = Σ_{nm} Iʲ_{nm}(x) · V_{(nm),ℓ}.

For flat tensor-product square panels with Lagrange basis at the GL nodes,
the values-to-coefficients map `V` is the identity, and the moments
collapse to the upsampling formula already used here. The two tasks
expressed in this language are:

- **Task 1** — restructure so that, for each source panel `j`, the
  geometry, upsampled nodes, and the "ws · Ex⊗Ex · scale" tensor
  (which is exactly the discrete moments-to-nodal map) are computed
  once and reused across every near target on every neighboring target
  panel.

- **Task 2** — for touching pairs (where smooth-quadrature moments are
  inaccurate), evaluate the moments by adaptive quadtree subdivision on
  `[-1,1]²` instead.

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

- Self-interaction `i == j` (handled separately; DT vanishes for flat
  same-panel pairs anyway).
- Singularity-aware initial subdivision (the v1 adaptive scheme is pure
  error-driven; biased initial subdivision can be added later).
- 2D Laplace and Helmholtz kernels.
- Curved patches.
- Replacing the upsampling path entirely (the design keeps both tiers).

## 4. Architecture

```
build_neighbor_list(..., correct_edges)
        │
        ▼
Dict{(i,j) => NearMethod}     # method.kind ∈ {:upsample, :adaptive}
        │
        ▼
_laplace3d_corrections(interface, dict, mode)
   for source panel i with ≥1 neighbor:
     if any neighbor of i is :upsample:
        build SourceCache_i   # p_up, Mt (moments-to-nodal)
     for each neighbor j of i:
        method = dict[(i,j)]
        if :upsample : K_block[t,:] = Mt * kvec(t)
        if :adaptive : K_block[t,:] = adaptive_panel_moments(panel_i, t, ...)
        K_direct  = laplace3d_{DT,D}_panel(panel_i, panel_j)
        push  (K_block - K_direct) into per-thread sparse triplets
   merge → SparseMatrixCSC
```

Key invariants:

- Both tiers produce a block of shape `(num_points(panel_j), n_quad²)`,
  so the downstream "subtract K_direct, push triplets" code is one
  branch.
- Parallelism is over source panels, not pairs — each thread owns its
  `SourceCache`. Per-thread scratch (`Dw`, `tb`, `bb`, plus
  adaptive-path stacks) sized for the largest source panel encountered.
- `correct_edges = false` (the default) preserves current behavior up
  to floating-point summation order: no `:adaptive` entries are
  emitted, edge panels remain filtered exactly as today, and the
  sparse correction matrix matches the current implementation to
  roundoff (see §8 test 1 for the precise tolerance).

## 5. Components

### 5.1 `NearMethod` value type (new)

In `src/kernel/laplace3d_near.jl`:

```julia
struct NearMethod
    kind::Symbol            # :upsample or :adaptive
    n_up::Int               # used when kind == :upsample
    atol::Float64           # used when kind == :adaptive
    n_GL::Int               # base GL order per leaf, when :adaptive
    max_depth::Int          # safety cap on subdivision, when :adaptive
end
```

`build_neighbor_list` returns `Dict{Tuple{Int,Int}, NearMethod}`.

### 5.2 `build_neighbor_list` (modified)

New kwarg `correct_edges::Bool = false`. Pair classification:

- `i == j` → skip (as today).
- coplanar same-plane (DT-vanishing test as today) → skip.
- if `correct_edges == false`:
    - any panel with `is_edge == true` → skip (as today).
    - otherwise emit `NearMethod(:upsample, n_up, …)`.
- if `correct_edges == true`:
    - `is_edge` filter dropped.
    - touching test: any corner of `i` within
      `tol_geom = √eps(T) · max(L_i, L_j)` of any corner of `j` →
      emit `NearMethod(:adaptive, …)`.
    - otherwise emit `NearMethod(:upsample, …)`.

The existing `check_quad_order3d`-driven `n_up` selection is preserved
for the `:upsample` branch.

### 5.3 `SourceCache{T}` (new)

In `src/kernel/laplace3d_near_upsampling.jl`:

```julia
struct SourceCache{T}
    panel::FlatPanel{T,3}
    n_up::Int
    p_up::Vector{NTuple{3,T}}   # length n_up²; upsampled physical positions
    Mt::Matrix{T}               # n_quad² × n_up²;
                                # Mt[m, α] = scale · ws_up[i] · ws_up[j]
                                #           · Ex[i, m_x] · Ex[j, m_y]
                                # where α = (i,j), m = (m_x, m_y) flattened
end

build_source_cache(panel, n_up, ns_up, ws_up, Ex) :: SourceCache
```

For each near target `t` on a `:upsample` pair:

```
kvec[α] = K(t, p_up[α])          # length n_up², depends on target
K_block[t, :] = Mt * kvec        # BLAS-2; or stack t's for BLAS-3
```

This is the discrete moments form with `V = I` (Lagrange basis at the
nodes). `Mt` is the "moments-to-nodal" tensor pre-folded with the
quadrature weights — built once per source panel.

### 5.4 `adaptive_panel_moments` (new)

New file `src/kernel/laplace3d_near_adaptive.jl`:

```julia
function adaptive_panel_moments_inplace!(
    K_row::AbstractVector{T},        # length n_quad², output
    panel_src::FlatPanel{T,3},
    point_trg::NTuple{3,T},
    trg_normal::NTuple{3,T},         # used for :DT; ignored for :D
    mode::Symbol,
    n_GL::Int,
    atol::T,
    max_depth::Int,
) where T
```

Recursive quadtree on `(u,v) ∈ [-1,1]²`:

1. Compute parent moments via `n_GL × n_GL` GL on the current cell.
2. Compute children moments on each of 4 sub-cells with the same rule.
3. `err = maximum(abs.(parent .- sum(children)))` across all `n_quad²`
   moments.
4. If `err < atol` or `depth == max_depth`: return `sum(children)`.
5. Else recurse on each child with `atol' = atol / 4` (or `atol`; choice
   recorded in `Open Question O1`).

At each GL node `(u, v)` on a leaf cell:

- physical point `y = X^j(u, v)`,
- kernel `K(point_trg, y)` (mode-dispatched: DT vs D),
- Lagrange basis values `Lm_x(u)`, `Lm_y(v)` via `barycentric_row!`
  using the parent panel's `gl_xs` and `bary_weights`,
- Jacobian factor `scale_leaf = (cell_size_u / 2) · (cell_size_v / 2)
  · scale_panel`.

Output `K_row[m]` accumulates `weight · K · L_{m_x}(u) · L_{m_y}(v) ·
scale_leaf` over all leaf GL nodes.

### 5.5 `_laplace3d_corrections` (restructured)

Outer loop becomes per source panel; inner loop is per neighbor target
panel. Branches on `method.kind`. Per-thread sparse-triplet buffers are
unchanged. `K_direct` subtraction unchanged.

### 5.6 Public API

`laplace3d_DT_fmm3d_corrected` and `laplace3d_D_fmm3d_corrected` gain
kwargs:

- `correct_edges::Bool = false`
- `adaptive_atol::Float64 = up_tol` (defaults to the existing up_tol)
- `adaptive_n_GL::Int = 0` (`0` → use the source panel's own `n_quad`)
- `adaptive_max_depth::Int = 20`

Signatures stay backward-compatible: omitting all four reproduces
current behavior.

## 6. Data Flow

Assembly time, single source panel `i`:

1. Look up all `(i, j)` entries in the dict.
2. If any has `kind == :upsample`, build `SourceCache_i` with the
   largest `n_up` requested by any such entry.
3. Compute `K_direct` for each `(i, j)` (existing routine).
4. For each `(i, j)`:
   - `:upsample` branch:
       per target `t` on panel `j`: evaluate `kvec[α] = K(t, p_up[α])`,
       `K_block[t, :] = Mt[m=1:n_quad², :] * kvec`.
   - `:adaptive` branch:
       per target `t`: call `adaptive_panel_moments_inplace!` into a
       row of `K_block`.
5. Push `(K_block − K_direct)` entries into thread-local sparse
   triplets.

Matvec time:

```
u = laplace3d_{DT,D}_fmm3d(...) · σ  +  corrections_sparse · σ
```

unchanged.

## 7. Error Handling

- **Adaptive non-convergence at `max_depth`.** Emit a single
  `@warn` per `(i, j)` (deduplicate via a thread-local `Set`), accept
  the deepest estimate, continue.
- **Touching test tolerance.** `tol_geom = √eps(T) · max(L_i, L_j)`
  is used. Box-corner geometries produce exact float matches between
  shared corners (same arithmetic path through `cc + bma·x + dma·y`),
  so this is generous.
- **DT-vanishing skip** runs before the touching test. Two coplanar
  touching panels still produce no entry — their moments are
  identically zero.
- **`correct_edges = true` with no touching pairs** in the geometry —
  legal; the dict simply contains no `:adaptive` entries.

## 8. Testing

1. **Refactor regression.** With `correct_edges = false`, the sparse
   correction matrix from the new code must agree with the current
   code to roundoff (Frobenius-norm relative diff < `1e-12` in
   `Float64`) on the existing test cases. Add a direct equality test
   against a frozen reference matrix on a small dielectric box.

2. **`SourceCache` reuse.** Instrument `build_source_cache` with an
   integer counter behind a debug flag and assert in a test that it
   is called exactly once per source panel that has at least one
   `:upsample` neighbor.

3. **Adaptive moments unit test.** Fixed source panel, single
   off-panel target. Compute each of the `n_quad²` moments with
   `adaptive_panel_moments_inplace!` and with high-order `hcubature`
   from `HCubature.jl`. Assert agreement to `atol`.

4. **Cross-face touching regression.** Build a 2-panel
   box-corner geometry (two perpendicular squares sharing an edge).
   Choose a smooth density. Compare DT applied to it via:
   (a) brute-force `hcubature` over both panels (reference),
   (b) new corrections with `correct_edges = false`,
   (c) new corrections with `correct_edges = true`.
   Assert that (c) is at least 4 orders of magnitude closer to (a)
   than (b) at moderate `atol` (e.g. `1e-8`).

5. **End-to-end dielectric box.** Rerun the dielectric-box example
   with `correct_edges = true`. Confirm that the GMRES residual at
   convergence and/or the solution error against a known analytic
   case (point-charge inside a dielectric box) drops compared to the
   `correct_edges = false` baseline. This is the headline acceptance
   check.

## 9. Open Questions

- **O1. Atol splitting in recursion.** Pass `atol/4` to children
  (additive splitting, classic h-adaptive) or pass `atol` unchanged
  (looser, faster, sometimes-pessimistic)? Start with `atol/4`;
  revisit if convergence is too aggressive.

## 10. Files Touched

- `src/kernel/laplace3d_near.jl` — `build_neighbor_list` signature
  and pair classification; thread the new kwargs through
  `laplace3d_DT_fmm3d_corrected` / `laplace3d_D_fmm3d_corrected`.
- `src/kernel/laplace3d_near_upsampling.jl` — introduce
  `SourceCache`, `build_source_cache`; restructure
  `_laplace3d_corrections`.
- `src/kernel/laplace3d_near_adaptive.jl` — new file with
  `adaptive_panel_moments_inplace!`.
- `src/BoundaryIntegral.jl` — `include` the new file.
- `test/` — three new test files for items 1–4 above; extend
  the dielectric-box example test for item 5.

## 11. Acceptance Criteria

- All existing tests pass unchanged with `correct_edges = false`
  (default).
- New tests (items 1–5 in §8) pass.
- Wall-time for assembly on a representative dielectric-box case
  with `correct_edges = false` is no slower than current, and ideally
  10–30% faster from `SourceCache` reuse.
- The dielectric-box example with `correct_edges = true` shows a
  measurable solution-error reduction over the `false` baseline.
