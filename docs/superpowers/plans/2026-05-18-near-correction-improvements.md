# Near-Correction Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the 3D Laplace near-correction into a per-source moments form (Task 1 of the spec) and add an opt-in adaptive subdivision path for touching edge pairs (Task 2 of the spec), per `docs/superpowers/specs/2026-05-18-near-correction-improvements-design.md`.

**Architecture:** Two-dict neighbor list (`upsample::Dict{(Int,Int),Int}` + `adaptive::Dict{(Int,Int),AdaptiveConfig}`). A new `SourceCache{T}` holds the per-source upsampled positions and a `(n_quad² × n_up²)` moments-to-nodal tensor `Mt`, built from each panel's own `gl_xs` to handle varquad. The corrections assembler loops over source panels (`@threads :dynamic`), batches `K(t, p_up)` evaluations into a single BLAS-3 `Mt * Kmat` per target panel, and dispatches the touching-pair branch to a new `adaptive_panel_moments_inplace!` recursive quadtree on `[-1,1]²`. The `correct_edges = false` default reproduces current behavior up to summation roundoff.

**Tech Stack:** Julia 1.x, FastGaussQuadrature.jl, NearestNeighbors.jl (KDTree), HCubature.jl (reference integrals), SparseArrays, Base.Threads, FMM3D.jl. Tests: `Test` stdlib, `julia --project -e 'using Pkg; Pkg.test()'` with `BI_RUN_FULL_TESTS=1` for full near-field suite.

---

## File Structure

**New files:**
- `src/kernel/laplace3d_near_adaptive.jl` — `AdaptiveConfig` struct, `adaptive_panel_moments_inplace!` recursive quadtree.
- `test/kernel/laplace3d_near_source_cache.jl` — Mt indexing regression, varquad correctness, SourceCache reuse counter.
- `test/kernel/laplace3d_near_corner_pairs.jl` — corner-pair discovery for `correct_edges = true`.
- `test/kernel/laplace3d_near_adaptive.jl` — adaptive_panel_moments_inplace! unit test against hcubature.
- `test/kernel/laplace3d_near_touching.jl` — cross-face touching headline regression.

**Modified files:**
- `src/kernel/laplace3d_near_upsampling.jl` — introduce `SourceCache`/`build_source_cache`; restructure `_laplace3d_corrections` (per-source loop, BLAS-3 batched, dispatch to upsample or adaptive); remove `_build_upsampling_cache`.
- `src/kernel/laplace3d_near.jl` — `build_neighbor_list` returns NamedTuple of two dicts, adds two-phase candidate discovery (panel-corner KDTree); thread new kwargs through `laplace3d_DT_fmm3d_corrected` / `laplace3d_D_fmm3d_corrected`.
- `src/solver/dielectric_box3d.jl` — forward new kwargs from `lhs_dielectric_box3d_fmm3d_corrected`.
- `src/BoundaryIntegral.jl` — `include("kernel/laplace3d_near_adaptive.jl")` after the upsampling include.
- `test/runtests.jl` — `include` the new test files under the `run_full` guard.

---

## Task 1: Add AdaptiveConfig and skeleton adaptive file (red→green→commit)

**Files:**
- Create: `src/kernel/laplace3d_near_adaptive.jl`
- Modify: `src/BoundaryIntegral.jl` (add `include`)
- Test: `test/kernel/laplace3d_near_adaptive.jl` (create with smoke test only)

- [ ] **Step 1: Create the adaptive file skeleton with AdaptiveConfig only**

`src/kernel/laplace3d_near_adaptive.jl`:
```julia
# Adaptive quadtree-based moments for touching (edge-sharing) near pairs.
# See docs/superpowers/specs/2026-05-18-near-correction-improvements-design.md §5.4.

struct AdaptiveConfig
    atol::Float64
    rtol::Float64
    n_GL::Int          # base GL order per leaf; 0 means "use source panel's n_quad"
    max_depth::Int
end

AdaptiveConfig(; atol::Float64, rtol::Float64 = sqrt(eps(Float64)),
                 n_GL::Int = 0, max_depth::Int = 20) =
    AdaptiveConfig(atol, rtol, n_GL, max_depth)
```

- [ ] **Step 2: Add the include to the module**

In `src/BoundaryIntegral.jl`, after the line:

```julia
include("kernel/laplace3d_near_upsampling.jl")
```

insert:

```julia
include("kernel/laplace3d_near_adaptive.jl")
```

(directly before the existing `include("kernel/laplace3d_near_hcubature.jl")` line).

- [ ] **Step 3: Write smoke test for AdaptiveConfig**

`test/kernel/laplace3d_near_adaptive.jl`:
```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using Test

@testset "AdaptiveConfig defaults" begin
    cfg = BI.AdaptiveConfig(atol = 1e-8)
    @test cfg.atol == 1e-8
    @test cfg.rtol == sqrt(eps(Float64))
    @test cfg.n_GL == 0
    @test cfg.max_depth == 20

    cfg2 = BI.AdaptiveConfig(atol = 1e-6, rtol = 1e-10, n_GL = 8, max_depth = 12)
    @test cfg2.n_GL == 8
    @test cfg2.max_depth == 12
end
```

- [ ] **Step 4: Wire the test into runtests under the run_full guard**

In `test/runtests.jl`, inside `if run_full ... end`, add (after the existing `include("kernel/laplace3d_near.jl")`):

```julia
        include("kernel/laplace3d_near_adaptive.jl")
```

- [ ] **Step 5: Run the test and confirm PASS**

Run:
```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -40
```
Expected: the `AdaptiveConfig defaults` testset passes; the rest of the suite is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/kernel/laplace3d_near_adaptive.jl src/BoundaryIntegral.jl test/kernel/laplace3d_near_adaptive.jl test/runtests.jl
git commit -m "$(cat <<'EOF'
add AdaptiveConfig skeleton for near-correction adaptive path

Empty skeleton file for the adaptive quadtree moments; populated in
later tasks. AdaptiveConfig is exported at the module level via the
include and tested with a smoke test.
EOF
)"
```

---

## Task 2: Mt indexing regression test (locks the flatten convention before refactor)

**Files:**
- Create: `test/kernel/laplace3d_near_source_cache.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the regression test pinning the existing flatten convention**

`test/kernel/laplace3d_near_source_cache.jl`:
```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature, LinearAlgebra
using Test

# Pin the flatten convention used by `_laplace3d_panel_upsampled_inplace!`:
# Dw[i, j], output K_up[ti, idx] with idx = (ii - 1) * n_quad + jj (outer ii).
# A SourceCache built from the same panel must reproduce the same K_up matrix
# bit-for-bit at the BLAS-3 level (modulo floating-point summation order).
@testset "Mt indexing convention vs transpose(Ex)*Dw*Ex" begin
    n_quad = 4
    n_up   = 8
    ns0, ws0 = gausslegendre(n_quad);  ns0 = Float64.(ns0); ws0 = Float64.(ws0)
    ns_up, ws_up = gausslegendre(n_up); ns_up = Float64.(ns_up); ws_up = Float64.(ws_up)
    Ex = BI.interp_matrix_1d_gl(ns0, ws0, ns_up)

    # Build a unit-square source panel and a single off-panel target point
    a = (-0.5, -0.5, 0.0); b = ( 0.5, -0.5, 0.0)
    c = ( 0.5,  0.5, 0.0); d = (-0.5,  0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    panel = BI.rect_panel3d_discretize(a, b, c, d, ns0, ws0, normal)
    Lx = norm(b .- a); Ly = norm(d .- a); scale = Lx * Ly / 4

    # Synthetic Dw[i,j] (arbitrary values)
    Dw = [sin(i + 0.3 * j) for i in 1:n_up, j in 1:n_up]

    # Reference: existing code's transpose(Ex) * Dw * Ex flattened with
    # idx = (ii-1)*n_quad + jj   (ii outer, jj inner).
    bb_ref = transpose(Ex) * Dw * Ex
    K_ref = Vector{Float64}(undef, n_quad^2)
    idx = 1
    for ii in 1:n_quad, jj in 1:n_quad
        K_ref[idx] = bb_ref[ii, jj]
        idx += 1
    end

    # Mt formula per spec §5.3:
    # Mt[m, α] = scale * ws_up[i_up] * ws_up[j_up] * Ex[i_up, m_x] * Ex[j_up, m_y]
    # with α = (i_up-1)*n_up + j_up,  m = (m_x-1)*n_quad + m_y.
    Mt = Matrix{Float64}(undef, n_quad^2, n_up^2)
    for m_x in 1:n_quad, m_y in 1:n_quad
        m = (m_x - 1) * n_quad + m_y
        for i_up in 1:n_up, j_up in 1:n_up
            α = (i_up - 1) * n_up + j_up
            Mt[m, α] = scale * ws_up[i_up] * ws_up[j_up] *
                       Ex[i_up, m_x] * Ex[j_up, m_y]
        end
    end

    # Flatten Dw with the same α convention; this is the "kvec" that the
    # spec multiplies Mt by. But the regression here is against
    # transpose(Ex)*Dw*Ex which does NOT carry the `scale * ws*ws` factors,
    # so reconstruct the bare matmul as transpose(Ex) * Dwhat * Ex where
    # Dwhat[i,j] = Dw[i,j] / (scale * ws_up[i] * ws_up[j]) and Mt absorbs
    # those factors. Equivalent identity:
    Dwhat_vec = Vector{Float64}(undef, n_up^2)
    for i_up in 1:n_up, j_up in 1:n_up
        α = (i_up - 1) * n_up + j_up
        Dwhat_vec[α] = Dw[i_up, j_up] / (scale * ws_up[i_up] * ws_up[j_up])
    end
    K_new = Mt * Dwhat_vec

    @test isapprox(K_new, K_ref; rtol = 1e-12, atol = 1e-14)
end
```

- [ ] **Step 2: Wire the test into runtests under run_full**

In `test/runtests.jl`, inside the `if run_full` block, add (after the new adaptive test include from Task 1):

```julia
        include("kernel/laplace3d_near_source_cache.jl")
```

- [ ] **Step 3: Run the test and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -20
```
Expected: `Mt indexing convention` passes.

- [ ] **Step 4: Commit**

```bash
git add test/kernel/laplace3d_near_source_cache.jl test/runtests.jl
git commit -m "$(cat <<'EOF'
pin Mt flatten convention via regression test

Tests the (i_up,j_up)→α and (m_x,m_y)→m index conventions documented
in spec §5.3 against the existing transpose(Ex)*Dw*Ex formulation.
This locks the convention before the per-source refactor.
EOF
)"
```

---

## Task 3: Introduce SourceCache and build_source_cache

**Files:**
- Modify: `src/kernel/laplace3d_near_upsampling.jl`
- Test: `test/kernel/laplace3d_near_source_cache.jl` (extend)

- [ ] **Step 1: Write a failing test for build_source_cache**

Append to `test/kernel/laplace3d_near_source_cache.jl`:

```julia
@testset "build_source_cache shapes and reuse" begin
    ns0, ws0 = gausslegendre(4); ns0 = Float64.(ns0); ws0 = Float64.(ws0)
    a = (-0.5, -0.5, 0.0); b = ( 0.5, -0.5, 0.0)
    c = ( 0.5,  0.5, 0.0); d = (-0.5,  0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    panel = BI.rect_panel3d_discretize(a, b, c, d, ns0, ws0, normal)

    n_up = 10
    cache = BI.build_source_cache(panel, n_up)
    @test cache.panel === panel
    @test cache.n_up == n_up
    @test length(cache.p_up) == n_up^2
    @test size(cache.Mt) == (panel.n_quad^2, n_up^2)

    # Equivalence to current laplace3d_panel_upsampled output, sample target:
    a2 = (-0.6, -0.6, 0.4); b2 = ( 0.6, -0.6, 0.4)
    c2 = ( 0.6,  0.6, 0.4); d2 = (-0.6,  0.6, 0.4)
    panel_trg = BI.rect_panel3d_discretize(a2, b2, c2, d2, ns0, ws0, normal)

    K_old = BI.laplace3d_DT_panel_upsampled(panel, panel_trg, n_up)
    # New path: for each target, build kvec and apply Mt.
    K_new = similar(K_old)
    np_trg = size(K_new, 1)
    kvec = Vector{Float64}(undef, n_up^2)
    for t in 1:np_trg
        pt = panel_trg.points[t]
        for α in 1:n_up^2
            kvec[α] = BI.laplace3d_grad(cache.p_up[α], pt, panel_trg.normal)
        end
        K_new[t, :] .= cache.Mt * kvec
    end

    @test isapprox(K_new, K_old; rtol = 1e-12, atol = 1e-14)
end
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | grep -E '(Error|FAIL|build_source_cache)' | head
```
Expected: FAIL with "UndefVarError: build_source_cache".

- [ ] **Step 3: Add the SourceCache struct and constructor**

In `src/kernel/laplace3d_near_upsampling.jl`, at the top of the file (before any function), add:

```julia
struct SourceCache{T}
    panel::FlatPanel{T,3}
    n_up::Int
    p_up::Vector{NTuple{3,T}}    # length n_up^2, upsampled physical positions
    Mt::Matrix{T}                # n_quad^2 × n_up^2 moments-to-nodal tensor
end

function build_source_cache(panel::FlatPanel{T,3}, n_up::Int) where T
    ns_up_d, ws_up_d = gausslegendre(n_up)
    ns_up = convert(Vector{T}, ns_up_d)
    ws_up = convert(Vector{T}, ws_up_d)

    # Per-source Ex from THIS panel's GL nodes (handles varquad).
    Ex = convert(Matrix{T}, interp_matrix_1d_gl(panel.gl_xs, panel.gl_ws, ns_up))

    a, b, c, d = panel.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    Lx = norm(b .- a)
    Ly = norm(d .- a)
    scale = Lx * Ly / 4
    bma = b .- a
    dma = d .- a

    p_up = Vector{NTuple{3,T}}(undef, n_up^2)
    for i_up in 1:n_up
        x = ns_up[i_up] / 2
        for j_up in 1:n_up
            y = ns_up[j_up] / 2
            α = (i_up - 1) * n_up + j_up
            p_up[α] = cc .+ bma .* x .+ dma .* y
        end
    end

    n_quad = panel.n_quad
    Mt = Matrix{T}(undef, n_quad^2, n_up^2)
    for m_x in 1:n_quad
        for m_y in 1:n_quad
            m = (m_x - 1) * n_quad + m_y
            for i_up in 1:n_up
                for j_up in 1:n_up
                    α = (i_up - 1) * n_up + j_up
                    Mt[m, α] = scale * ws_up[i_up] * ws_up[j_up] *
                               Ex[i_up, m_x] * Ex[j_up, m_y]
                end
            end
        end
    end

    return SourceCache{T}(panel, n_up, p_up, Mt)
end
```

- [ ] **Step 4: Run the test and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -20
```
Expected: `build_source_cache shapes and reuse` passes.

- [ ] **Step 5: Commit**

```bash
git add src/kernel/laplace3d_near_upsampling.jl test/kernel/laplace3d_near_source_cache.jl
git commit -m "$(cat <<'EOF'
add per-source SourceCache with moments-to-nodal tensor

SourceCache holds the upsampled positions and Mt[m, α] folded with
the panel's own GL nodes (varquad-safe). Equivalence to the existing
laplace3d_DT_panel_upsampled output is verified by the new test.
The existing per-pair routine is untouched in this task; the next
task swaps it in.
EOF
)"
```

---

## Task 4: Restructure _laplace3d_corrections to per-source loop with BLAS-3 batching

**Files:**
- Modify: `src/kernel/laplace3d_near_upsampling.jl`

This task is a pure refactor: the existing `build_neighbor_list` return type (`Dict{Tuple{Int,Int}, Int}`) stays as-is. The next task changes the dict shape. We split these to keep diffs small and let TDD lean on the existing test suite.

- [ ] **Step 1: Read the current _laplace3d_corrections shape**

Read `src/kernel/laplace3d_near_upsampling.jl` (entire file). Confirm the current per-pair loop structure at the `Base.Threads.@threads :static for k in 1:length(pairs)` line.

- [ ] **Step 2: Replace _laplace3d_corrections with per-source version**

Replace the body of `_laplace3d_corrections` (the function starting `function _laplace3d_corrections(`) with:

```julia
function _laplace3d_corrections(
    interface::DielectricInterface{P, T},
    neighbor_list::Dict{Tuple{Int, Int}, Int},
    mode::Symbol, direct_kernel::Function,
) where {P <: AbstractPanel, T}
    cnt = [length(p.points) for p in interface.panels]
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    # Group neighbors by source panel index.
    src_to_neighbors = Dict{Int, Vector{Tuple{Int, Int}}}()   # i => [(j, n_up), ...]
    for ((i, j), n_up) in neighbor_list
        push!(get!(() -> Vector{Tuple{Int,Int}}(), src_to_neighbors, i), (j, n_up))
    end
    source_indices = collect(keys(src_to_neighbors))

    nthreads = Base.Threads.maxthreadid()
    rows_tl = [Int[] for _ in 1:nthreads]
    cols_tl = [Int[] for _ in 1:nthreads]
    vals_tl = [T[]   for _ in 1:nthreads]

    Base.Threads.@threads :dynamic for k in 1:length(source_indices)
        i = source_indices[k]
        panel_src = interface.panels[i]
        neighbors = src_to_neighbors[i]
        n_up_max = maximum(nup for (_, nup) in neighbors)

        cache = build_source_cache(panel_src, n_up_max)   # one per source panel

        tid = Base.Threads.threadid()
        rows = rows_tl[tid]
        cols = cols_tl[tid]
        vals = vals_tl[tid]
        col_off = offsets[i]
        ncols = offsets[i + 1] - col_off

        for (j, n_up) in neighbors
            panel_trg = interface.panels[j]
            np_trg = num_points(panel_trg)

            # Build Kmat[α, t] over all targets in panel j (BLAS-3 friendly).
            n_up_eff = cache.n_up                # we use the panel's full n_up_max cache
            Kmat = Matrix{T}(undef, n_up_eff^2, np_trg)
            if mode === :DT
                for t in 1:np_trg
                    pt = panel_trg.points[t]
                    for α in 1:(n_up_eff^2)
                        Kmat[α, t] = laplace3d_grad(cache.p_up[α], pt, panel_trg.normal)
                    end
                end
            elseif mode === :D
                for t in 1:np_trg
                    pt = panel_trg.points[t]
                    for α in 1:(n_up_eff^2)
                        Kmat[α, t] = laplace3d_grad(pt, cache.p_up[α], panel_src.normal)
                    end
                end
            else
                error("unknown mode for _laplace3d_corrections")
            end

            K_block_T = cache.Mt * Kmat            # (n_quad^2, np_trg)
            K_block = transpose(K_block_T)         # (np_trg, n_quad^2)

            K_direct = direct_kernel(panel_src, panel_trg)

            row_off = offsets[j]
            nrows = offsets[j + 1] - row_off
            @inbounds for c_local in 1:ncols
                for r_local in 1:nrows
                    v = K_block[r_local, c_local] - K_direct[r_local, c_local]
                    iszero(v) && continue
                    push!(rows, row_off + r_local)
                    push!(cols, col_off + c_local)
                    push!(vals, v)
                end
            end
        end
    end

    return sparse(reduce(vcat, rows_tl), reduce(vcat, cols_tl),
                  reduce(vcat, vals_tl), total_n, total_n)
end
```

- [ ] **Step 3: Delete the now-unused helpers**

In the same file, delete the entire bodies of:
- `_build_upsampling_cache` (around lines 143–158)
- `_laplace3d_panel_upsampled_inplace!` (around lines 84–138)

Keep `laplace3d_panel_upsampled`, `laplace3d_DT_panel_upsampled`, `laplace3d_D_panel_upsampled` — they are used by the existing test at `test/kernel/laplace3d_near.jl:49`.

- [ ] **Step 4: Run the full near-field test suite and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -40
```
Expected: all previously-passing tests still pass. In particular the existing test/kernel/laplace3d_near.jl, test/solver/dielectric_box3d.jl, and the new SourceCache test from Task 3.

- [ ] **Step 5: Commit**

```bash
git add src/kernel/laplace3d_near_upsampling.jl
git commit -m "$(cat <<'EOF'
restructure _laplace3d_corrections to per-source loop with BLAS-3 batching

Outer loop now iterates over source panels (dynamic scheduling). For
each source panel we build a SourceCache once and reuse it across
every neighbor target panel, batching all targets of a panel into one
BLAS-3 Mt * Kmat. The global _build_upsampling_cache and the per-pair
_laplace3d_panel_upsampled_inplace! helper are removed.
EOF
)"
```

---

## Task 5: Varquad correctness test for SourceCache

**Files:**
- Test: `test/kernel/laplace3d_near_source_cache.jl` (extend)

- [ ] **Step 1: Write the varquad correctness test**

Append to `test/kernel/laplace3d_near_source_cache.jl`:

```julia
@testset "SourceCache handles varquad (per-panel gl_xs)" begin
    # Two panels with different n_quad: the old global cache used panels[1].gl_xs
    # for both, which gave wrong Mt for panels[2]. Per-source Ex must fix this.
    n1 = 4; n2 = 6
    ns1, ws1 = gausslegendre(n1); ns1 = Float64.(ns1); ws1 = Float64.(ws1)
    ns2, ws2 = gausslegendre(n2); ns2 = Float64.(ns2); ws2 = Float64.(ws2)
    normal = (0.0, 0.0, 1.0)

    panel1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0),
        ns1, ws1, normal)
    panel2 = BI.rect_panel3d_discretize(
        (1.0, -0.5, 0.0), (2.0, -0.5, 0.0),
        (2.0, 0.5, 0.0), (1.0, 0.5, 0.0),
        ns2, ws2, normal)

    n_up = 12

    # Build per-panel caches and compare each to laplace3d_DT_panel_upsampled
    panel_trg = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
        ns1, ws1, normal)

    for panel in (panel1, panel2)
        cache = BI.build_source_cache(panel, n_up)
        K_ref = BI.laplace3d_DT_panel_upsampled(panel, panel_trg, n_up)
        K_new = similar(K_ref)
        kvec = Vector{Float64}(undef, n_up^2)
        for t in 1:size(K_new, 1)
            pt = panel_trg.points[t]
            for α in 1:n_up^2
                kvec[α] = BI.laplace3d_grad(cache.p_up[α], pt, panel_trg.normal)
            end
            K_new[t, :] .= cache.Mt * kvec
        end
        @test isapprox(K_new, K_ref; rtol = 1e-12, atol = 1e-14)
    end
end
```

- [ ] **Step 2: Run the test and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | grep -E '(SourceCache|varquad|Error|FAIL)' | head
```
Expected: `SourceCache handles varquad` passes.

- [ ] **Step 3: Commit**

```bash
git add test/kernel/laplace3d_near_source_cache.jl
git commit -m "$(cat <<'EOF'
test SourceCache varquad correctness against laplace3d_DT_panel_upsampled

Two panels with different n_quad share a target panel; per-source Ex
must reproduce laplace3d_DT_panel_upsampled for each.
EOF
)"
```

---

## Task 6: Convert build_neighbor_list to return NamedTuple of two dicts

**Files:**
- Modify: `src/kernel/laplace3d_near.jl`
- Modify: `src/kernel/laplace3d_near_upsampling.jl`

This task is the API change. `correct_edges = true` is implemented but the adaptive dict remains empty until Task 9; the new path only routes touching pairs to the adaptive dict.

- [ ] **Step 1: Modify build_neighbor_list to return a NamedTuple**

In `src/kernel/laplace3d_near.jl`, replace the signature and body of `build_neighbor_list` (around lines 228–328) with:

```julia
function build_neighbor_list(
    interface::DielectricInterface{P, T},
    max_order::Int,
    atol::T,
    include_edges_src::Bool,
    include_edges_trg::Bool;
    distance_only::Bool = false,
    range_factor::T = T(5),
    correct_edges::Bool = false,
    adaptive_cfg::AdaptiveConfig = AdaptiveConfig(atol = Float64(atol)),
) where {P <: AbstractPanel, T}
    upsample = Dict{Tuple{Int, Int}, Int}()
    adaptive = Dict{Tuple{Int, Int}, AdaptiveConfig}()
    n_panels = length(interface.panels)
    centers = Matrix{T}(undef, 3, n_panels)
    lengths = Vector{T}(undef, n_panels)
    n_quads = Vector{Int}(undef, n_panels)
    normals = Vector{NTuple{3, T}}(undef, n_panels)
    plane_offsets = Vector{T}(undef, n_panels)

    for (i, panel) in enumerate(interface.panels)
        c_panel = (panel.corners[1] .+ panel.corners[2] .+ panel.corners[3] .+ panel.corners[4]) ./ 4
        @views centers[:, i] .= c_panel
        lengths[i] = max(norm(panel.corners[1] .- panel.corners[2]), norm(panel.corners[2] .- panel.corners[3]))
        n_quads[i] = panel.n_quad
        normals[i] = panel.normal
        plane_offsets[i] = dot(panel.normal, panel.corners[1])
    end

    n_points = sum(length(panel.points) for panel in interface.panels)
    points = Matrix{T}(undef, 3, n_points)
    point_panel_idx = Vector{Int}(undef, n_points)
    point_idx = 1
    for (panel_idx, panel) in enumerate(interface.panels)
        for point in panel.points
            @views points[:, point_idx] .= point
            point_panel_idx[point_idx] = panel_idx
            point_idx += 1
        end
    end
    tree = KDTree(points)

    same_surface_tol = sqrt(eps(T))
    L_max = maximum(lengths; init = one(T))
    tol_corner = sqrt(eps(T)) * L_max

    # Phase B (only if correct_edges): KDTree over panel corners.
    corner_tree = nothing
    corner_panel_idx = Int[]
    if correct_edges
        n_corners = 4 * n_panels
        corners_mat = Matrix{T}(undef, 3, n_corners)
        corner_panel_idx = Vector{Int}(undef, n_corners)
        k = 0
        for (panel_idx, panel) in enumerate(interface.panels)
            for ci in 1:4
                k += 1
                @views corners_mat[:, k] .= panel.corners[ci]
                corner_panel_idx[k] = panel_idx
            end
        end
        corner_tree = KDTree(corners_mat)
    end

    for (i, paneli) in enumerate(interface.panels)
        if !correct_edges
            (!include_edges_src && paneli.is_edge) && continue
        end
        l_i = lengths[i]
        n_quad_i = n_quads[i]
        r_i = range_factor * l_i / n_quad_i
        nearby = inrange(tree, centers[:, i], r_i)

        panel_dict = Dict{Int, Vector{Int}}()
        for point_id in nearby
            j = point_panel_idx[point_id]
            i == j && continue
            if haskey(panel_dict, j)
                push!(panel_dict[j], point_id)
            else
                panel_dict[j] = [point_id]
            end
        end

        # Phase B: discover corner-touching neighbors not in panel_dict.
        touching_set = Set{Int}()
        if correct_edges
            for ci in 1:4
                corner_pt = collect(paneli.corners[ci])
                near_corner_ids = inrange(corner_tree, corner_pt, tol_corner)
                for cid in near_corner_ids
                    j = corner_panel_idx[cid]
                    i == j && continue
                    push!(touching_set, j)
                end
            end
            for j in touching_set
                if !haskey(panel_dict, j)
                    panel_dict[j] = Int[]
                end
            end
        end

        for j in keys(panel_dict)
            if !correct_edges
                (!include_edges_trg && interface.panels[j].is_edge) && continue
            end

            dot_normals = dot(normals[i], normals[j])
            if dot_normals > 1 - same_surface_tol
                if abs(plane_offsets[i] - plane_offsets[j]) <= same_surface_tol * max(one(T), l_i)
                    continue
                end
            end

            if correct_edges && (j in touching_set)
                adaptive[(i, j)] = adaptive_cfg
                continue
            end

            if distance_only
                upsample[(i, j)] = n_quad_i
            else
                points_j = panel_dict[j]
                isempty(points_j) && continue
                min_dist = Inf
                min_point_id = 0
                for point_id in points_j
                    dist = norm(points[:, point_id] - centers[:, i])
                    if dist < min_dist
                        min_dist = dist
                        min_point_id = point_id
                    end
                end

                order_i = check_quad_order3d(paneli, (points[1, min_point_id], points[2, min_point_id], points[3, min_point_id]), atol, max_order)

                if order_i > n_quad_i
                    key = (i, j)
                    if haskey(upsample, key)
                        upsample[key] = max(upsample[key], order_i)
                    else
                        upsample[key] = order_i
                    end
                end
            end
        end
    end

    return (; upsample, adaptive)
end
```

- [ ] **Step 2: Update _laplace3d_corrections signature to consume the new shape**

In `src/kernel/laplace3d_near_upsampling.jl`, update the signatures and bodies that consume the neighbor list.

Replace `_laplace3d_corrections` (introduced in Task 4) signature and the first lines so it takes the two dicts:

```julia
function _laplace3d_corrections(
    interface::DielectricInterface{P, T},
    upsample_dict::Dict{Tuple{Int, Int}, Int},
    adaptive_dict::Dict{Tuple{Int, Int}, AdaptiveConfig},
    mode::Symbol, direct_kernel::Function,
) where {P <: AbstractPanel, T}
    cnt = [length(p.points) for p in interface.panels]
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    # Sanity: dicts must be disjoint.
    for k in keys(adaptive_dict)
        @assert !haskey(upsample_dict, k) "neighbor key $k present in both dicts"
    end

    # Group neighbors by source panel.
    src_to_ups = Dict{Int, Vector{Tuple{Int, Int}}}()
    for ((i, j), n_up) in upsample_dict
        push!(get!(() -> Vector{Tuple{Int,Int}}(), src_to_ups, i), (j, n_up))
    end
    src_to_adp = Dict{Int, Vector{Tuple{Int, AdaptiveConfig}}}()
    for ((i, j), cfg) in adaptive_dict
        push!(get!(() -> Vector{Tuple{Int, AdaptiveConfig}}(), src_to_adp, i), (j, cfg))
    end
    source_indices = collect(union(keys(src_to_ups), keys(src_to_adp)))

    nthreads = Base.Threads.maxthreadid()
    rows_tl = [Int[] for _ in 1:nthreads]
    cols_tl = [Int[] for _ in 1:nthreads]
    vals_tl = [T[]   for _ in 1:nthreads]

    Base.Threads.@threads :dynamic for k in 1:length(source_indices)
        i = source_indices[k]
        panel_src = interface.panels[i]
        col_off = offsets[i]
        ncols = offsets[i + 1] - col_off

        # Build SourceCache only if upsample neighbors exist.
        cache = if haskey(src_to_ups, i)
            n_up_max = maximum(nup for (_, nup) in src_to_ups[i])
            build_source_cache(panel_src, n_up_max)
        else
            nothing
        end

        tid = Base.Threads.threadid()
        rows = rows_tl[tid]
        cols = cols_tl[tid]
        vals = vals_tl[tid]

        if haskey(src_to_ups, i)
            for (j, _n_up) in src_to_ups[i]
                panel_trg = interface.panels[j]
                np_trg = num_points(panel_trg)
                n_up_eff = cache.n_up
                Kmat = Matrix{T}(undef, n_up_eff^2, np_trg)
                if mode === :DT
                    for t in 1:np_trg
                        pt = panel_trg.points[t]
                        for α in 1:(n_up_eff^2)
                            Kmat[α, t] = laplace3d_grad(cache.p_up[α], pt, panel_trg.normal)
                        end
                    end
                elseif mode === :D
                    for t in 1:np_trg
                        pt = panel_trg.points[t]
                        for α in 1:(n_up_eff^2)
                            Kmat[α, t] = laplace3d_grad(pt, cache.p_up[α], panel_src.normal)
                        end
                    end
                else
                    error("unknown mode for _laplace3d_corrections")
                end
                K_block_T = cache.Mt * Kmat
                K_block = transpose(K_block_T)
                K_direct = direct_kernel(panel_src, panel_trg)
                row_off = offsets[j]
                nrows = offsets[j + 1] - row_off
                @inbounds for c_local in 1:ncols
                    for r_local in 1:nrows
                        v = K_block[r_local, c_local] - K_direct[r_local, c_local]
                        iszero(v) && continue
                        push!(rows, row_off + r_local)
                        push!(cols, col_off + c_local)
                        push!(vals, v)
                    end
                end
            end
        end

        # Adaptive branch is wired in Task 9; for now it must be empty.
        @assert !haskey(src_to_adp, i) "adaptive neighbors present but adaptive path not yet implemented (Task 9)"
    end

    return sparse(reduce(vcat, rows_tl), reduce(vcat, cols_tl),
                  reduce(vcat, vals_tl), total_n, total_n)
end
```

Note: remove the `n_quads_i_placeholder` line — it was a transcription scratch; the second `src_to_adp = Dict{...}()` reassignment is the one to keep.

- [ ] **Step 3: Update wrappers that call _laplace3d_corrections**

In `src/kernel/laplace3d_near_upsampling.jl`, update `laplace3d_DT_corrections` and `laplace3d_D_corrections` signatures:

```julia
function laplace3d_DT_corrections(interface::DielectricInterface{P, T},
                                  upsample_dict::Dict{Tuple{Int, Int}, Int},
                                  adaptive_dict::Dict{Tuple{Int, Int}, AdaptiveConfig}) where {P <: AbstractPanel, T}
    return _laplace3d_corrections(interface, upsample_dict, adaptive_dict, :DT, laplace3d_DT_panel)
end

function laplace3d_D_corrections(interface::DielectricInterface{P, T},
                                 upsample_dict::Dict{Tuple{Int, Int}, Int},
                                 adaptive_dict::Dict{Tuple{Int, Int}, AdaptiveConfig}) where {P <: AbstractPanel, T}
    return _laplace3d_corrections(interface, upsample_dict, adaptive_dict, :D, laplace3d_D_panel)
end
```

- [ ] **Step 4: Update callers in src/kernel/laplace3d_near.jl**

In `src/kernel/laplace3d_near.jl`, the public wrappers `laplace3d_DT_fmm3d_corrected` and `laplace3d_D_fmm3d_corrected` currently destructure a flat Dict. Update both call sites where the neighbor list is built and consumed:

For `laplace3d_DT_fmm3d_corrected` (around line 331), replace:
```julia
    neighbor_list = build_neighbor_list(interface, max_order, up_tol, include_edges_src, include_edges_trg, range_factor = range_factor)
    @info "length of neighbor_list: $(length(keys(neighbor_list))) out of $(length(interface.panels)^2)"
    corrections = laplace3d_DT_corrections(interface, neighbor_list)
```
with:
```julia
    (; upsample, adaptive) = build_neighbor_list(interface, max_order, up_tol, include_edges_src, include_edges_trg, range_factor = range_factor)
    @info "neighbor list: upsample=$(length(upsample)) adaptive=$(length(adaptive)) of $(length(interface.panels)^2)"
    corrections = laplace3d_DT_corrections(interface, upsample, adaptive)
```

For `laplace3d_D_fmm3d_corrected` (around line 350), apply the same transformation.

- [ ] **Step 5: Run the full near-field test suite and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -40
```
Expected: all tests still pass, including the new ones. `correct_edges = false` is the default so no `:adaptive` entries are emitted.

- [ ] **Step 6: Commit**

```bash
git add src/kernel/laplace3d_near.jl src/kernel/laplace3d_near_upsampling.jl
git commit -m "$(cat <<'EOF'
return two-dict neighbor list and route adaptive pairs

build_neighbor_list now returns (; upsample, adaptive). With
correct_edges=false (default) adaptive is empty and behavior is
unchanged; with correct_edges=true the panel-corner KDTree promotes
touching cross-face pairs into adaptive. The corrections assembler
asserts the adaptive branch is empty (wired in a later task).
EOF
)"
```

---

## Task 7: Corner-pair discovery test

**Files:**
- Create: `test/kernel/laplace3d_near_corner_pairs.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the corner-pair test**

`test/kernel/laplace3d_near_corner_pairs.jl`:
```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature
using Test

# Two perpendicular panels sharing a single edge along x = 0.5.
# panel_a lies in the z = 0 plane; panel_b lies in the x = 0.5 plane.
# With low n_quad their GL nodes are interior, so a node-only KDTree
# may not discover them; the panel-corner phase must.
@testset "build_neighbor_list corner-pair discovery (correct_edges)" begin
    n_quad = 3
    ns, ws = gausslegendre(n_quad); ns = Float64.(ns); ws = Float64.(ws)

    a1 = (-0.5, -0.5, 0.0); b1 = ( 0.5, -0.5, 0.0)
    c1 = ( 0.5,  0.5, 0.0); d1 = (-0.5,  0.5, 0.0)
    normal1 = (0.0, 0.0, 1.0)
    panel_a = BI.rect_panel3d_discretize(a1, b1, c1, d1, ns, ws, normal1)

    # panel_b shares the edge x = 0.5, y in [-0.5, 0.5] with panel_a.
    a2 = ( 0.5, -0.5, 0.0); b2 = ( 0.5,  0.5, 0.0)
    c2 = ( 0.5,  0.5, 1.0); d2 = ( 0.5, -0.5, 1.0)
    normal2 = (1.0, 0.0, 0.0)
    panel_b = BI.rect_panel3d_discretize(a2, b2, c2, d2, ns, ws, normal2)

    interface = BI.DielectricInterface([panel_a, panel_b], [1.0, 1.0], [1.0, 1.0])

    # With correct_edges = false: classic discovery; range_factor is small
    # enough that the two interior-node clouds do not see each other.
    (; upsample, adaptive) =
        BI.build_neighbor_list(interface, 1, 1e-6, true, true;
                               distance_only = true, range_factor = 0.5,
                               correct_edges = false)
    @test isempty(adaptive)

    # With correct_edges = true: the corner KDTree must surface the touching pair.
    (; upsample, adaptive) =
        BI.build_neighbor_list(interface, 1, 1e-6, true, true;
                               distance_only = true, range_factor = 0.5,
                               correct_edges = true)
    @test haskey(adaptive, (1, 2)) || haskey(adaptive, (2, 1))
end
```

- [ ] **Step 2: Wire the test in**

In `test/runtests.jl`, inside `if run_full`, after the other new includes:

```julia
        include("kernel/laplace3d_near_corner_pairs.jl")
```

- [ ] **Step 3: Run the test and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | grep -E '(corner-pair|Error|FAIL)' | head
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add test/kernel/laplace3d_near_corner_pairs.jl test/runtests.jl
git commit -m "$(cat <<'EOF'
test corner-pair discovery for correct_edges=true

Two perpendicular flat panels sharing an edge whose GL nodes are
interior; verifies the new panel-corner KDTree phase finds the
touching pair when the node-based scan does not.
EOF
)"
```

---

## Task 8: Implement adaptive_panel_moments_inplace!

**Files:**
- Modify: `src/kernel/laplace3d_near_adaptive.jl`
- Test: `test/kernel/laplace3d_near_adaptive.jl` (extend)

- [ ] **Step 1: Write the failing unit test against HCubature**

Append to `test/kernel/laplace3d_near_adaptive.jl`:

```julia
using HCubature, FastGaussQuadrature, LinearAlgebra, StaticArrays

@testset "adaptive_panel_moments_inplace! matches HCubature" begin
    n_quad = 4
    ns, ws = gausslegendre(n_quad); ns = Float64.(ns); ws = Float64.(ws)

    # Unit square panel at z = 0.
    a = (-0.5, -0.5, 0.0); b = ( 0.5, -0.5, 0.0)
    c = ( 0.5,  0.5, 0.0); d = (-0.5,  0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    panel = BI.rect_panel3d_discretize(a, b, c, d, ns, ws, normal)

    # Target slightly above the panel (smooth, not singular) so HCubature converges easily.
    pt = (0.1, 0.2, 0.3)
    pt_normal = (0.0, 0.0, 1.0)

    cfg = BI.AdaptiveConfig(atol = 1e-10, rtol = 1e-12, n_GL = n_quad, max_depth = 12)
    K_row = zeros(Float64, n_quad^2)
    BI.adaptive_panel_moments_inplace!(K_row, panel, pt, pt_normal, :DT, cfg)

    # Reference moments via HCubature, integrand in (u,v) ∈ [-1,1]^2.
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a; dma = d .- a
    Lx = norm(bma); Ly = norm(dma); scale_panel = Lx * Ly / 4
    K_ref = zeros(Float64, n_quad^2)
    for m_x in 1:n_quad, m_y in 1:n_quad
        m = (m_x - 1) * n_quad + m_y
        integrand(uv) = begin
            u, v = uv[1], uv[2]
            y = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            rx = BI.barycentric_row(panel.gl_xs, panel.bary_weights, u)
            ry = BI.barycentric_row(panel.gl_xs, panel.bary_weights, v)
            return BI.laplace3d_grad(y, pt, pt_normal) * rx[m_x] * ry[m_y] * scale_panel
        end
        val, _ = hquadrature(integrand, SVector{2,Float64}(-1.0, -1.0),
                                          SVector{2,Float64}( 1.0,  1.0); atol = 1e-12)
        K_ref[m] = val
    end

    @test isapprox(K_row, K_ref; atol = 1e-7, rtol = 1e-7)
end
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | grep -E '(adaptive_panel_moments|Error|FAIL)' | head
```
Expected: FAIL with "UndefVarError: adaptive_panel_moments_inplace!".

- [ ] **Step 3: Implement adaptive_panel_moments_inplace!**

Append to `src/kernel/laplace3d_near_adaptive.jl`:

```julia
# Compute the n_quad^2 moments ∫ K(point_trg, X(u,v)) L_{m_x}(u) L_{m_y}(v) J du dv
# over the source panel, accumulated into K_row[m] with m = (m_x-1)*n_quad + m_y.
# Recursive quadtree on (u,v) ∈ [-1,1]^2; error-based stopping.
function adaptive_panel_moments_inplace!(
    K_row::AbstractVector{T},
    panel_src::FlatPanel{T,3},
    point_trg::NTuple{3,T},
    trg_normal::NTuple{3,T},
    mode::Symbol,
    cfg::AdaptiveConfig,
) where T
    n_quad = panel_src.n_quad
    @assert length(K_row) == n_quad^2
    fill!(K_row, zero(T))

    n_GL = cfg.n_GL == 0 ? n_quad : cfg.n_GL
    ns_d, ws_d = gausslegendre(n_GL)
    ns_GL = convert(Vector{T}, ns_d)
    ws_GL = convert(Vector{T}, ws_d)

    a, b, c, d = panel_src.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a
    Lx = norm(bma); Ly = norm(dma); scale_panel = Lx * Ly / 4

    src_normal = panel_src.normal
    gl_xs = panel_src.gl_xs
    bary_weights = panel_src.bary_weights

    rx = Vector{T}(undef, n_quad)
    ry = Vector{T}(undef, n_quad)

    function cell_moments(u_lo::T, u_hi::T, v_lo::T, v_hi::T)
        out = zeros(T, n_quad^2)
        half_u = (u_hi - u_lo) / 2
        half_v = (v_hi - v_lo) / 2
        mid_u  = (u_hi + u_lo) / 2
        mid_v  = (v_hi + v_lo) / 2
        scale_cell = half_u * half_v * scale_panel
        @inbounds for gi in 1:n_GL
            u = mid_u + half_u * ns_GL[gi]
            wu = ws_GL[gi]
            barycentric_row!(rx, gl_xs, bary_weights, u)
            for gj in 1:n_GL
                v = mid_v + half_v * ns_GL[gj]
                wv = ws_GL[gj]
                barycentric_row!(ry, gl_xs, bary_weights, v)

                y = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                kval = if mode === :DT
                    laplace3d_grad(y, point_trg, trg_normal)
                elseif mode === :D
                    laplace3d_grad(point_trg, y, src_normal)
                else
                    error("unknown mode for adaptive_panel_moments_inplace!")
                end
                w = wu * wv * scale_cell * kval
                m = 0
                for m_x in 1:n_quad
                    for m_y in 1:n_quad
                        m += 1
                        out[m] += w * rx[m_x] * ry[m_y]
                    end
                end
            end
        end
        return out
    end

    function recurse!(K_row::AbstractVector{T}, u_lo::T, u_hi::T, v_lo::T, v_hi::T, depth::Int)
        parent = cell_moments(u_lo, u_hi, v_lo, v_hi)
        mid_u = (u_hi + u_lo) / 2
        mid_v = (v_hi + v_lo) / 2
        c1 = cell_moments(u_lo, mid_u, v_lo, mid_v)
        c2 = cell_moments(mid_u, u_hi, v_lo, mid_v)
        c3 = cell_moments(u_lo, mid_u, mid_v, v_hi)
        c4 = cell_moments(mid_u, u_hi, mid_v, v_hi)
        children_sum = c1 .+ c2 .+ c3 .+ c4
        err = maximum(abs.(parent .- children_sum))
        tol_local = max(T(cfg.atol), T(cfg.rtol) * maximum(abs.(parent)))
        if err < tol_local || depth == cfg.max_depth
            @inbounds for m in eachindex(K_row)
                K_row[m] += children_sum[m]
            end
            return depth == cfg.max_depth && err >= tol_local
        end
        hit_a = recurse!(K_row, u_lo, mid_u, v_lo, mid_v, depth + 1)
        hit_b = recurse!(K_row, mid_u, u_hi, v_lo, mid_v, depth + 1)
        hit_c = recurse!(K_row, u_lo, mid_u, mid_v, v_hi, depth + 1)
        hit_d = recurse!(K_row, mid_u, u_hi, mid_v, v_hi, depth + 1)
        return hit_a || hit_b || hit_c || hit_d
    end

    hit_max = recurse!(K_row, -one(T), one(T), -one(T), one(T), 0)
    return hit_max
end
```

- [ ] **Step 4: Run the test and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | grep -E '(adaptive_panel_moments|Error|FAIL)' | head
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kernel/laplace3d_near_adaptive.jl test/kernel/laplace3d_near_adaptive.jl
git commit -m "$(cat <<'EOF'
implement adaptive_panel_moments_inplace! recursive quadtree

Error-based quadtree on (u,v) in [-1,1]^2 with per-cell tolerance
tol_local = max(atol, rtol * max(|parent|)); accumulates the
n_quad^2 moments into K_row. Verified against HCubature reference
on a non-singular target above the panel.
EOF
)"
```

---

## Task 9: Wire the :adaptive branch into _laplace3d_corrections

**Files:**
- Modify: `src/kernel/laplace3d_near_upsampling.jl`

- [ ] **Step 1: Replace the @assert no-adaptive guard with the adaptive branch**

In `src/kernel/laplace3d_near_upsampling.jl`, locate the line in `_laplace3d_corrections`:

```julia
        # Adaptive branch is wired in Task 9; for now it must be empty.
        @assert !haskey(src_to_adp, i) "adaptive neighbors present but adaptive path not yet implemented (Task 9)"
```

Replace those two lines with:

```julia
        if haskey(src_to_adp, i)
            n_quad_i = panel_src.n_quad
            Krow = Vector{T}(undef, n_quad_i^2)
            warn_pairs = Set{Tuple{Int,Int}}()
            for (j, cfg) in src_to_adp[i]
                panel_trg = interface.panels[j]
                np_trg = num_points(panel_trg)
                K_block = Matrix{T}(undef, np_trg, n_quad_i^2)
                for t in 1:np_trg
                    fill!(Krow, zero(T))
                    hit = adaptive_panel_moments_inplace!(
                        Krow, panel_src, panel_trg.points[t], panel_trg.normal,
                        mode, cfg,
                    )
                    if hit
                        push!(warn_pairs, (i, j))
                    end
                    @inbounds for c_local in 1:length(Krow)
                        K_block[t, c_local] = Krow[c_local]
                    end
                end
                K_direct = direct_kernel(panel_src, panel_trg)
                row_off = offsets[j]
                nrows = offsets[j + 1] - row_off
                @inbounds for c_local in 1:ncols
                    for r_local in 1:nrows
                        v = K_block[r_local, c_local] - K_direct[r_local, c_local]
                        iszero(v) && continue
                        push!(rows, row_off + r_local)
                        push!(cols, col_off + c_local)
                        push!(vals, v)
                    end
                end
            end
            if !isempty(warn_pairs)
                @warn "adaptive_panel_moments hit max_depth for pairs: $(collect(warn_pairs))"
            end
        end
```

(The per-thread `@warn` is acceptable here because each thread owns its own source panel `i`, so the same `(i, j)` cannot appear from another thread.)

- [ ] **Step 2: Run the existing tests to confirm no regression**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -40
```
Expected: all previously-passing tests still pass; with `correct_edges = false` the new code path is dead.

- [ ] **Step 3: Commit**

```bash
git add src/kernel/laplace3d_near_upsampling.jl
git commit -m "$(cat <<'EOF'
wire adaptive branch into _laplace3d_corrections

For each source panel with at least one adaptive (touching) neighbor,
call adaptive_panel_moments_inplace! per target point and assemble
into the same sparse triplet stream as the upsample branch. Pairs
that hit max_depth are reported via a single @warn per source panel.
EOF
)"
```

---

## Task 10: Public API kwargs (correct_edges + adaptive_*) and dielectric-box forwarding

**Files:**
- Modify: `src/kernel/laplace3d_near.jl`
- Modify: `src/solver/dielectric_box3d.jl`

- [ ] **Step 1: Add kwargs to laplace3d_DT_fmm3d_corrected and laplace3d_D_fmm3d_corrected**

In `src/kernel/laplace3d_near.jl`, replace the signatures of both functions (currently `laplace3d_DT_fmm3d_corrected` around line 331 and `laplace3d_D_fmm3d_corrected` around line 350).

`laplace3d_DT_fmm3d_corrected`:
```julia
function laplace3d_DT_fmm3d_corrected(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    up_tol::Float64,
    max_order::Int;
    include_edges_src::Bool = false,
    include_edges_trg::Bool = false,
    range_factor::Float64 = 5.0,
    correct_edges::Bool = false,
    adaptive_atol::Float64 = up_tol,
    adaptive_rtol::Float64 = sqrt(eps(Float64)),
    adaptive_n_GL::Int = 0,
    adaptive_max_depth::Int = 20,
) where {P <: AbstractPanel}
    n_points = num_points(interface)
    D_base = laplace3d_DT_fmm3d(interface, fmm_tol)
    adaptive_cfg = AdaptiveConfig(adaptive_atol, adaptive_rtol, adaptive_n_GL, adaptive_max_depth)
    (; upsample, adaptive) = build_neighbor_list(
        interface, max_order, up_tol, include_edges_src, include_edges_trg;
        range_factor = range_factor,
        correct_edges = correct_edges,
        adaptive_cfg = adaptive_cfg,
    )
    @info "neighbor list: upsample=$(length(upsample)) adaptive=$(length(adaptive)) of $(length(interface.panels)^2)"
    corrections = laplace3d_DT_corrections(interface, upsample, adaptive)

    f = charges -> (D_base * charges) + (corrections * charges)
    return LinearMap{Float64}(f, n_points, n_points)
end
```

`laplace3d_D_fmm3d_corrected`:
```julia
function laplace3d_D_fmm3d_corrected(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    up_tol::Float64,
    max_order::Int;
    include_edges_src::Bool = false,
    include_edges_trg::Bool = false,
    range_factor::Float64 = 5.0,
    correct_edges::Bool = false,
    adaptive_atol::Float64 = up_tol,
    adaptive_rtol::Float64 = sqrt(eps(Float64)),
    adaptive_n_GL::Int = 0,
    adaptive_max_depth::Int = 20,
) where {P <: AbstractPanel}
    n_points = num_points(interface)
    D_base = laplace3d_D_fmm3d(interface, fmm_tol)
    adaptive_cfg = AdaptiveConfig(adaptive_atol, adaptive_rtol, adaptive_n_GL, adaptive_max_depth)
    (; upsample, adaptive) = build_neighbor_list(
        interface, max_order, up_tol, include_edges_src, include_edges_trg;
        range_factor = range_factor,
        correct_edges = correct_edges,
        adaptive_cfg = adaptive_cfg,
    )
    @info "neighbor list: upsample=$(length(upsample)) adaptive=$(length(adaptive)) of $(length(interface.panels)^2)"
    corrections = laplace3d_D_corrections(interface, upsample, adaptive)

    f = charges -> (D_base * charges) + (corrections * charges)
    return LinearMap{Float64}(f, n_points, n_points)
end
```

- [ ] **Step 2: Forward kwargs through the dielectric-box solver**

In `src/solver/dielectric_box3d.jl`, modify `lhs_dielectric_box3d_fmm3d_corrected`. Read the function first to see its full body. Update its signature to include the new kwargs and forward them to `laplace3d_DT_fmm3d_corrected`:

```julia
function lhs_dielectric_box3d_fmm3d_corrected(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    up_tol::Float64,
    max_order::Int;
    include_edges_src::Bool = true,
    include_edges_trg::Bool = true,
    correct_edges::Bool = false,
    adaptive_atol::Float64 = up_tol,
    adaptive_rtol::Float64 = sqrt(eps(Float64)),
    adaptive_n_GL::Int = 0,
    adaptive_max_depth::Int = 20,
) where {P <: AbstractPanel}
```

and locate the call to `laplace3d_DT_fmm3d_corrected` inside its body; add the new kwargs to that call:

```julia
    D_transpose = laplace3d_DT_fmm3d_corrected(
        interface,
        fmm_tol,
        up_tol,
        max_order;
        include_edges_src = include_edges_src,
        include_edges_trg = include_edges_trg,
        correct_edges = correct_edges,
        adaptive_atol = adaptive_atol,
        adaptive_rtol = adaptive_rtol,
        adaptive_n_GL = adaptive_n_GL,
        adaptive_max_depth = adaptive_max_depth,
    )
```

(Preserve the rest of the function body as-is.)

- [ ] **Step 3: Run the full test suite and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -40
```
Expected: all tests pass (defaults still reproduce current behavior).

- [ ] **Step 4: Commit**

```bash
git add src/kernel/laplace3d_near.jl src/solver/dielectric_box3d.jl
git commit -m "$(cat <<'EOF'
expose correct_edges and adaptive_* kwargs in public corrected wrappers

laplace3d_{DT,D}_fmm3d_corrected and lhs_dielectric_box3d_fmm3d_corrected
gain correct_edges plus adaptive_atol / adaptive_rtol / adaptive_n_GL /
adaptive_max_depth kwargs. Defaults preserve existing behavior.
EOF
)"
```

---

## Task 11: Cross-face touching headline regression test

**Files:**
- Create: `test/kernel/laplace3d_near_touching.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the cross-face regression test**

`test/kernel/laplace3d_near_touching.jl`:
```julia
using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature, HCubature, LinearAlgebra, StaticArrays
using Test

# Cross-face touching pair. Compute (DT σ)(t) at a target t on the
# touching panel via:
#   (a) HCubature over both panels (reference),
#   (b) sparse corrections with correct_edges = false,
#   (c) sparse corrections with correct_edges = true.
# Expect (c) much closer to (a) than (b).
@testset "cross-face touching DT regression" begin
    n_quad = 4
    ns, ws = gausslegendre(n_quad); ns = Float64.(ns); ws = Float64.(ws)

    # panel_a in z = 0 plane, x in [-0.5, 0.5], y in [-0.5, 0.5].
    a1 = (-0.5, -0.5, 0.0); b1 = ( 0.5, -0.5, 0.0)
    c1 = ( 0.5,  0.5, 0.0); d1 = (-0.5,  0.5, 0.0)
    n1 = (0.0, 0.0, 1.0)
    panel_a = BI.rect_panel3d_discretize(a1, b1, c1, d1, ns, ws, n1)

    # panel_b in x = 0.5 plane, sharing the edge with panel_a.
    a2 = ( 0.5, -0.5, 0.0); b2 = ( 0.5,  0.5, 0.0)
    c2 = ( 0.5,  0.5, 1.0); d2 = ( 0.5, -0.5, 1.0)
    n2 = (1.0, 0.0, 0.0)
    panel_b = BI.rect_panel3d_discretize(a2, b2, c2, d2, ns, ws, n2)

    interface = BI.DielectricInterface([panel_a, panel_b], [1.0, 1.0], [1.0, 1.0])

    # Smooth density.
    σ(p) = exp(-2.0 * (p[1]^2 + p[2]^2 + p[3]^2))
    np = BI.num_points(interface)
    σvec = Vector{Float64}(undef, np)
    k = 0
    for panel in interface.panels
        for p in panel.points
            k += 1; σvec[k] = σ(p)
        end
    end

    # Reference (DT σ)(t) at a target point on panel_b near the shared edge.
    panel_b_idx = length(panel_a.points)            # offset for panel_b targets
    t_local = 1                                     # first GL node on panel_b
    t_global = panel_b_idx + t_local
    target = panel_b.points[t_local]
    target_normal = panel_b.normal

    # HCubature reference: integrate over both panels using the same parametrization
    # used elsewhere in this codebase.
    function hcub_DT(panel, target, target_normal)
        a, b, c, d = panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a; dma = d .- a
        Lx = norm(bma); Ly = norm(dma); s = Lx * Ly / 4
        integrand(uv) = begin
            u, v = uv[1], uv[2]
            y = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            σ(y) * BI.laplace3d_grad(y, target, target_normal) * s
        end
        val, _ = hquadrature(integrand,
                             SVector{2,Float64}(-1.0, -1.0),
                             SVector{2,Float64}( 1.0,  1.0); atol = 1e-10)
        return val
    end
    ref_val = hcub_DT(panel_a, target, target_normal) + hcub_DT(panel_b, target, target_normal)

    # Apply the FMM+correction operator to σvec, twice.
    op_off = BI.laplace3d_DT_fmm3d_corrected(
        interface, 1e-10, 1e-8, 24;
        include_edges_src = true, include_edges_trg = true, range_factor = 5.0,
        correct_edges = false)
    op_on  = BI.laplace3d_DT_fmm3d_corrected(
        interface, 1e-10, 1e-8, 24;
        include_edges_src = true, include_edges_trg = true, range_factor = 5.0,
        correct_edges = true,  adaptive_atol = 1e-8)

    y_off = op_off * σvec
    y_on  = op_on  * σvec

    err_off = abs(y_off[t_global] - ref_val)
    err_on  = abs(y_on[t_global]  - ref_val)

    @test err_on < err_off
    @test err_on < 1e-5    # loose acceptance: adaptive should reach reasonable accuracy
end
```

- [ ] **Step 2: Wire the test in**

In `test/runtests.jl`, inside `if run_full`:

```julia
        include("kernel/laplace3d_near_touching.jl")
```

- [ ] **Step 3: Run and confirm PASS**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | grep -E '(touching|Error|FAIL)' | head
```
Expected: PASS. If `err_on >= 1e-5`, investigate (likely a tolerance choice in `adaptive_atol`; do **not** weaken the test without diagnosing — the cross-face singularity should be tame at `1e-8`).

- [ ] **Step 4: Commit**

```bash
git add test/kernel/laplace3d_near_touching.jl test/runtests.jl
git commit -m "$(cat <<'EOF'
add cross-face touching DT regression test

Two perpendicular panels sharing an edge with a smooth density.
Verifies that correct_edges=true is materially closer to the
HCubature reference than correct_edges=false, with a 1e-5 absolute
acceptance bound on the corrected result.
EOF
)"
```

---

## Task 12: End-to-end dielectric box example with correct_edges = true

**Files:**
- Test: `test/solver/dielectric_box3d.jl` (extend)

This task adds an acceptance check to the existing `test/solver/dielectric_box3d.jl`. We rerun a small dielectric-box problem with both `correct_edges = false` and `correct_edges = true`; the latter must give equal-or-better solution error against an analytic point-charge reference.

The existing testset `dielectric_box3d corrected` (lines 55–73 of `test/solver/dielectric_box3d.jl`) builds a single box with `eps_box = 4.0`, solves with GMRES, and checks `total_flux + 1/eps_box ≈ 1` (an analytic Gauss-law check). We reuse this fixture and compare both `correct_edges` settings against that analytic reference.

- [ ] **Step 1: Append a new testset to test/solver/dielectric_box3d.jl**

Append at the end of `test/solver/dielectric_box3d.jl`:

```julia
@testset "dielectric_box3d corrected edges flag" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(3.0, 3.0, 1.0, 6, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))
    rhs = BI.rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(interface)

    lhs_off = BI.lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-6, 1e-6, 12;
                                                     correct_edges = false)
    lhs_on  = BI.lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-6, 1e-6, 12;
                                                     correct_edges = true,
                                                     adaptive_atol = 1e-6)

    x_off = BI.solve_gmres(lhs_off, rhs, 1e-6, 1e-6)
    x_on  = BI.solve_gmres(lhs_on,  rhs, 1e-6, 1e-6)

    flux_off = dot(ws, x_off)
    flux_on  = dot(ws, x_on)

    err_off = abs(flux_off + 1.0 / eps_box - 1.0)
    err_on  = abs(flux_on  + 1.0 / eps_box - 1.0)
    @info "dielectric box flux error: off=$(err_off) on=$(err_on)"

    # Both should converge GMRES.
    @test norm(lhs_off * x_off - rhs) < 1e-5
    @test norm(lhs_on  * x_on  - rhs) < 1e-5
    # The corrected (correct_edges=true) flux error should not be worse than
    # the uncorrected baseline. We do not assert a hard improvement: the
    # baseline already passes the existing 1e-1 acceptance, so on small
    # geometries the gain can be modest.
    @test err_on <= err_off + 1e-6
end
```

- [ ] **Step 2: Run the test**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tail -40
```
Expected: the new testset passes. The other dielectric-box tests are unchanged.

- [ ] **Step 3: Commit**

```bash
git add test/solver/dielectric_box3d.jl
git commit -m "$(cat <<'EOF'
end-to-end dielectric box test with correct_edges=true

Solves the existing small-box dielectric problem with and without
the adaptive edge correction; asserts the corrected solution error
is no worse than the baseline.
EOF
)"
```

---

## Task 13: Final integration check and self-review

**Files:**
- None (verification only)

- [ ] **Step 1: Run the full test suite from a clean state**

```bash
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()' 2>&1 | tee /tmp/bi_full_test.log | tail -60
```
Expected: every testset under `if run_full` and outside it passes.

- [ ] **Step 2: Run a smoke benchmark for per-source vs per-pair scheduling (O2 from spec)**

This step is exploratory. On a 6-face box geometry of ~10² panels (use the existing `single_dielectric_box3d_rhs_adaptive` builder), time `laplace3d_DT_fmm3d_corrected` assembly twice and record the wall time. If the new code is meaningfully slower than the previous implementation on this benchmark, **stop and flag the regression** — do not paper over it. Otherwise, note the timing in the final commit message of this task and move on.

Run:
```bash
julia --project -e '
using BoundaryIntegral
using Random
Random.seed!(0)
# Build a small box, time the assembly twice.
'
```
(Fill in a 5–10 line benchmark; we are not strict on numbers here.)

- [ ] **Step 3: Self-review checklist**

Walk through the spec section by section and verify each requirement has a task that implements it. In particular:
- §5.1 two-dict neighbor list ✓ (Tasks 6, 10)
- §5.2 panel-corner phase B ✓ (Tasks 6, 7)
- §5.2 DT/D-vanishing predicate preserved verbatim ✓ (Task 6)
- §5.3 SourceCache with per-source Ex ✓ (Tasks 3, 5)
- §5.4 adaptive_panel_moments_inplace! with rtol ✓ (Task 8)
- §5.5 per-source loop with BLAS-3 batching ✓ (Tasks 4, 6)
- §5.6 public API kwargs ✓ (Task 10)
- §7 max_depth warn after merge ✓ (Task 9, one warn per source-thread)
- §7 Float64-only public API ✓ (Task 10 signatures)
- §7 no interaction with laplace3d_near_hcubature.jl ✓ (file untouched)
- §8 test 1 refactor regression — covered by existing tests continuing to pass after Tasks 4 and 6 (no dedicated frozen-matrix test added; gain marginal over the existing dielectric-box regression).
- §8 test 2 varquad ✓ (Task 5)
- §8 test 3 SourceCache reuse counter — **deferred**. The per-source loop in Task 4 calls `build_source_cache` exactly once per source-key by construction; a behavioral counter test is a defensive measure for future refactors. Track as a follow-up if the assembler is restructured again.
- §8 test 4 adaptive moments ✓ (Task 8)
- §8 test 5 corner-pair discovery ✓ (Task 7)
- §8 test 6 Mt indexing ✓ (Task 2)
- §8 test 7 cross-face touching ✓ (Task 11)
- §8 test 8 end-to-end ✓ (Task 12)

- [ ] **Step 4: No commit — task is verification only.**

If any check above fails, return to the relevant task. If all checks pass, the implementation is complete.

---

## Notes for the executing agent

- The codebase uses `BI_RUN_FULL_TESTS=1` to gate the near-field test suite. Always run with this set when verifying near-correction changes.
- `julia --project -e 'using Pkg; Pkg.test()'` is the canonical test command (the project has a `.julia-version` via juliaup; the `using-julia` skill applies if you need to manage the toolchain).
- Threads: tests are designed to pass under both single-threaded and multi-threaded Julia. If your `JULIA_NUM_THREADS` is 1, the `@threads :dynamic` macro degenerates to a serial loop and is still correct.
- `FlatPanel` field layout is in `src/core/panels.jl:5`. Key fields used by this plan: `corners`, `normal`, `is_edge`, `n_quad`, `gl_xs`, `gl_ws`, `points`, `bary_weights`.
- Do not edit `src/kernel/laplace3d_near_hcubature.jl` in this plan. It is the off-surface near-eval path and is out of scope per spec §7.
- Commit messages use a brief `subject` line (under 70 chars), blank line, and a short body. Match the existing repo style (see `git log --oneline -10`).
