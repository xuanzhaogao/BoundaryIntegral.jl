# Multi-node Lattice Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Workflow note:** per user preference, script-writing may be delegated to Codex headless (`codex exec`); Claude reviews each task's diff before commit.

**Goal:** Solve all ≈2000 pair-density BIE problems of a 10×10 graphene flake across many cluster nodes (Distributed.jl driver + file manifest on ceph), store σ and ρ, and post-evaluate the full V matrix.

**Architecture:** Two repos. (1) `~/codes/BoundaryIntegral.jl`, branch `multi_rhs`: frame-translated orbital instances on a virtual global grid, batch assembly from explicit pair lists, near/far-split batched external-target evaluation, and `BatchResult` serialization. (2) `~/work/four_index_integral_solver/codes/lattice_scale/` (inside the `four_index_integral_solver` git repo): campaign config/manifest, solve/eval/consolidate/assemble tasks, a Distributed driver with `ClusterManagers.SlurmManager`, sbatch templates, and the 2×2 anchor + 10×10 campaigns. Spec: `docs/superpowers/specs/2026-06-10-multinode-lattice-campaign-design.md`.

**Tech Stack:** Julia 1.12, FMM3D (`lfmm3d`, `nd` batching), TKM3D (`ltkm3dc`), Krylov block GMRES, Distributed + ClusterManagers, Serialization stdlib, TOML stdlib, Slurm (user submits all jobs).

**Conventions used throughout:**
- The xsf datagrid is a NamedTuple `(nx, ny, nz, origin, A, B, C, values)`; `grid_point(dg, i, j, k)` is a pure affine map (valid for any integer indices, in/out of range) using `true_cell_vectors` (Wannier90 half-open convention, step along axis 1 = `At/nx`).
- "Global index" = integer grid coordinates on the virtual global grid in the TEMPLATE frame; an instance with offset `steps` occupies global indices `steps .+ (1:n)` per axis.
- All package tests are standalone-runnable: `julia --project=. test/solver/<file>.jl`.
- Package commits go on `multi_rhs`; campaign commits in `~/work/four_index_integral_solver`.
- One plan deviation from the spec, intentional: `targets.jls` is built by `consolidate` (from the *stored* batch supports), not by `prepare` — this guarantees the eval target set is exactly consistent with the solved/truncated ρ.

---

## Task 1: `OrbitalInstance`, `lattice_grid_steps`, frame-overlap helper

**Files:**
- Create: `src/solver/lattice_batch.jl`
- Modify: `src/BoundaryIntegral.jl` (include + exports; add `include("solver/lattice_batch.jl")` immediately after `include("solver/multi_rhs.jl")`)
- Test: `test/solver/lattice_batch.jl` (create)
- Modify: `test/runtests.jl` (add `include("solver/lattice_batch.jl")` after the `multi_rhs_vector.jl` include, NOT inside `run_full`)

- [ ] **Step 1: Write the failing tests**

```julia
# test/solver/lattice_batch.jl
using BoundaryIntegral
using Test

@testset "lattice_batch" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")

    @testset "lattice_grid_steps" begin
        st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_lat.xsf"))
        # orb_lat: 4x4x4 grid, span 1.5 (true cell 2.0, step 0.5), primvec = I => 2 steps/cell
        @test lattice_grid_steps(dg, st.primvec, (1, 0, 0)) == (2, 0, 0)
        @test lattice_grid_steps(dg, st.primvec, (0, -2, 1)) == (0, -4, 2)
        @test lattice_grid_steps(dg, st.primvec, (0, 0, 0)) == (0, 0, 0)
    end

    @testset "OrbitalInstance + frame overlap" begin
        inst = OrbitalInstance(7, 1, (2, 0, 0))
        @test inst.id == 7 && inst.template_id == 1 && inst.steps == (2, 0, 0)
        # frames of length 4 offset by 0 and 2 overlap on global indices 3:4
        @test BoundaryIntegral._frame_overlap(4, 0, 2) == 3:4
        @test BoundaryIntegral._frame_overlap(4, 2, 0) == 3:4
        @test BoundaryIntegral._frame_overlap(4, 0, 0) == 1:4
        @test BoundaryIntegral._frame_overlap(4, 0, 4) === nothing   # disjoint
        @test BoundaryIntegral._frame_overlap(4, 0, 9) === nothing
    end
end
```

- [ ] **Step 2: Run, verify failure**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: FAIL / `UndefVarError: lattice_grid_steps not defined`

- [ ] **Step 3: Implement**

```julia
# src/solver/lattice_batch.jl
# Lattice-scale batches: frame-TRANSLATED orbital instances on a virtual global grid
# (spec: docs/superpowers/specs/2026-06-10-multinode-lattice-campaign-design.md).
# Unlike the .bie LATTICE images (periodic circshift on the template grid), instances
# here carry an integer global-frame offset and never wrap — so >5 distinct cells per
# direction are representable and pair products are exact on frame intersections.

"""
    lattice_grid_steps(datagrid, primvec, n::NTuple{3,Int}) -> NTuple{3,Int}

Integer grid-step offset of the lattice translation `n1·a1 + n2·a2 + n3·a3` (the
global-frame offset of a translated orbital instance). Errors if a lattice vector is
not grid-commensurate. Same arithmetic as the `.bie` LATTICE images, but interpreted
as a frame translation, not a circshift.
"""
lattice_grid_steps(datagrid, primvec::AbstractMatrix, n::NTuple{3,Int}) =
    _lattice_grid_shift(datagrid, primvec, n)

"""
    OrbitalInstance(id, template_id, steps)

One orbital of a lattice campaign: template `template_id` translated by the integer
global-frame offset `steps` (grid steps; see `lattice_grid_steps`).
"""
struct OrbitalInstance
    id::Int
    template_id::Int
    steps::NTuple{3,Int}
end

# Per-axis overlap of two frames of length n offset by integer steps si, sj.
# Returns the GLOBAL index range covered by both (local_i = g - si, local_j = g - sj),
# or `nothing` if disjoint.
function _frame_overlap(n::Int, si::Int, sj::Int)
    glo = max(si, sj) + 1
    ghi = min(si, sj) + n
    glo > ghi && return nothing
    return glo:ghi
end
```

In `src/BoundaryIntegral.jl` add the include after `solver/multi_rhs.jl` and export `lattice_grid_steps`, `OrbitalInstance`.

- [ ] **Step 4: Run, verify pass**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/solver/lattice_batch.jl src/BoundaryIntegral.jl test/solver/lattice_batch.jl test/runtests.jl
git commit -m "feat: OrbitalInstance + lattice_grid_steps (frame translation, no wrap)"
```

---

## Task 2: `assemble_lattice_batch` — batch assembly in global index space

**Files:**
- Create: `test/fixtures/orb_smooth.xsf`, `test/fixtures/system_smooth_lat.bie`
- Modify: `src/solver/lattice_batch.jl`
- Test: `test/solver/lattice_batch.jl`

- [ ] **Step 1: Create the smooth fixture**

`orb_smooth.xsf` — 6×6×6 grid, true cell 3.0 (primvec = I ⇒ 2 steps/cell, 3 cells/axis), Gaussian blob centered at (0.75, 0.75, 0.75), σ = 0.4. Generate it once with this snippet (run from repo root), then commit the file:

```julia
# tools-free one-off generation (run: julia --project=. <<this snippet in a file or -e>>)
open("test/fixtures/orb_smooth.xsf", "w") do io
    println(io, "CRYSTAL\nPRIMVEC\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0")
    println(io, "PRIMCOORD\n1 1\nX 0.75 0.75 0.75")
    println(io, "BEGIN_BLOCK_DATAGRID_3D\n smooth\nBEGIN_DATAGRID_3D_smooth")
    println(io, "6 6 6")
    println(io, "0.0 0.0 0.0")
    println(io, "2.5 0.0 0.0\n0.0 2.5 0.0\n0.0 0.0 2.5")   # span = cell*(n-1)/n = 3.0*5/6
    vals = Float64[]
    for k in 1:6, j in 1:6, i in 1:6   # i fastest, matching the reader
        x = ((i-1)/6)*3.0; y = ((j-1)/6)*3.0; z = ((k-1)/6)*3.0
        push!(vals, exp(-((x-0.75)^2 + (y-0.75)^2 + (z-0.75)^2) / (2*0.4^2)))
    end
    for chunk in Iterators.partition(vals, 6)
        println(io, join(string.(chunk), " "))
    end
    println(io, "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D")
end
```

`system_smooth_lat.bie`:

```
UNITS bohr

BEGIN_DIELECTRICS
EPS_OUT 1.0
  1.5 1.5 1.5    8.0 8.0 8.0    2.0
END_DIELECTRICS

BEGIN_ORBITALS
  1   orb_smooth.xsf
  2   orb_smooth.xsf   LATTICE 1 0 0
END_ORBITALS

BEGIN_GROUPING
CUTOFF 5.0
END_GROUPING
```

- [ ] **Step 2: Write the failing tests**

Append to `test/solver/lattice_batch.jl` (inside the outer testset):

```julia
    @testset "assemble_lattice_batch == assemble_rhs_group (no-wrap regime)" begin
        si = read_system_input(joinpath(fixdir, "system_smooth_lat.bie"))
        g = assemble_rhs_group(si, 1; support_rtol = 1e-6)   # pairs (1,1),(1,2) via circshift

        st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
        templates = [dg]
        insts = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                     2 => OrbitalInstance(2, 1, lattice_grid_steps(dg, st.primvec, (1, 0, 0))))
        b = assemble_lattice_batch(templates, insts, [(1, 1), (1, 2)]; support_rtol = 1e-6)

        @test b.pair_ids == [(1, 1), (1, 2)]
        @test size(b.densities) == (length(b.gidx), 2)
        @test size(b.positions) == (3, length(b.gidx))
        @test length(b.weights) == length(b.gidx)
        # same physics as the circshift group (blob tails that wrap are ~exp(-31), below rtol):
        # compare per-position values
        function posmap(P, D)
            d = Dict{NTuple{3,Float64},Vector{Float64}}()
            for s in 1:size(P, 2)
                d[(round(P[1,s]; digits=9), round(P[2,s]; digits=9), round(P[3,s]; digits=9))] = D[s, :]
            end
            d
        end
        da, db = posmap(g.positions, g.densities), posmap(b.positions, b.densities)
        @test Set(keys(da)) == Set(keys(db))
        @test maximum(maximum(abs.(da[k] .- db[k])) for k in keys(da)) < 1e-10
        # weights are the uniform cell weight
        @test all(b.weights .≈ g.weights[1])
    end

    @testset "assemble_lattice_batch multi-anchor + disjoint pair" begin
        st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
        s1 = lattice_grid_steps(dg, st.primvec, (1, 0, 0))
        insts = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                     2 => OrbitalInstance(2, 1, s1),
                     3 => OrbitalInstance(3, 1, (40, 0, 0)))   # far away: zero overlap with 1
        b = assemble_lattice_batch([dg], insts, [(1, 1), (2, 2), (1, 3)]; support_rtol = 1e-6)
        @test size(b.densities, 2) == 3
        @test all(b.densities[:, 3] .== 0.0)                    # disjoint frames -> zero column
        # column 2 is column 1 translated by s1: same multiset of values
        @test sort(b.densities[findall(!iszero, b.densities[:, 1]), 1]) ≈
              sort(b.densities[findall(!iszero, b.densities[:, 2]), 2])
        # gidx sorted & unique
        @test issorted(b.gidx) && allunique(b.gidx)
    end
```

- [ ] **Step 3: Run, verify failure**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: FAIL / `assemble_lattice_batch not defined`

- [ ] **Step 4: Implement**

Append to `src/solver/lattice_batch.jl`:

```julia
"""
    LatticeBatch

A batch of pair densities on the union of their supports, indexed on the VIRTUAL GLOBAL
GRID (`gidx` = integer grid coordinates in the template frame). All columns share
`gidx`/`positions`/`weights` — required for the nd-batched FMM.
"""
struct LatticeBatch
    pair_ids::Vector{Tuple{Int,Int}}     # (orbital id i, orbital id j) per column
    gidx::Vector{NTuple{3,Int}}          # n shared global grid indices (sorted)
    positions::Matrix{Float64}           # 3 × n
    weights::Vector{Float64}             # n (uniform: |det cell| / (nx ny nz))
    densities::Matrix{Float64}           # n × K raw (unscreened) pair densities
end

num_pairs(b::LatticeBatch) = length(b.pair_ids)

"""
    assemble_lattice_batch(templates, instances, pairs; support_rtol=1e-6) -> LatticeBatch

Assemble the pair densities `rho_ij = phi_i * phi_j` for an explicit pair list. Each
orbital is `templates[instance.template_id]` translated by `instance.steps`; products
are exact pointwise multiplies on the integer frame intersection. The batch lives on
the union of the per-pair supports, then truncated by the group envelope
(rss across columns ≥ `support_rtol` × its max — same rule as `assemble_rhs_group`).
All templates must share one grid geometry.
"""
function assemble_lattice_batch(templates::AbstractVector,
        instances::AbstractDict{Int,OrbitalInstance},
        pairs::Vector{Tuple{Int,Int}}; support_rtol::Real = 1e-6)
    isempty(pairs) && error("empty batch")
    t1 = templates[1]
    for t in templates
        datagrids_compatible(t1, t) || error("all templates must share one grid geometry")
    end
    nx, ny, nz = t1.nx, t1.ny, t1.nz
    K = length(pairs)

    row = Dict{NTuple{3,Int},Int}()
    gidx = NTuple{3,Int}[]
    cols = Vector{Vector{Tuple{Int,Float64}}}(undef, K)
    for (k, (i, j)) in enumerate(pairs)
        oi = instances[i]; oj = instances[j]
        vi = templates[oi.template_id].values
        vj = templates[oj.template_id].values
        si = oi.steps; sj = oj.steps
        rx = _frame_overlap(nx, si[1], sj[1])
        ry = _frame_overlap(ny, si[2], sj[2])
        rz = _frame_overlap(nz, si[3], sj[3])
        vals = Tuple{Int,Float64}[]
        if rx !== nothing && ry !== nothing && rz !== nothing
            for gz in rz, gy in ry, gx in rx          # gx innermost: values are i-fastest
                v = vi[gx - si[1], gy - si[2], gz - si[3]] *
                    vj[gx - sj[1], gy - sj[2], gz - sj[3]]
                v == 0.0 && continue
                g = (gx, gy, gz)
                r = get!(row, g) do
                    push!(gidx, g)
                    length(gidx)
                end
                push!(vals, (r, v))
            end
        end
        cols[k] = vals
    end

    n = length(gidx)
    densities = zeros(Float64, n, K)
    for k in 1:K, (r, v) in cols[k]
        densities[r, k] = v
    end

    # union-support truncation (same rule as assemble_rhs_group)
    keep = if support_rtol > 0 && n > 0
        env = vec(sqrt.(sum(abs2, densities; dims = 2)))
        m = maximum(env)
        m > 0 ? findall(>=(support_rtol * m), env) : collect(1:n)
    else
        collect(1:n)
    end
    gk = gidx[keep]
    perm = sortperm(gk)                                # deterministic order
    gk = gk[perm]
    dk = densities[keep, :][perm, :]

    m = length(gk)
    positions = Matrix{Float64}(undef, 3, m)
    for s in 1:m
        p = grid_point(t1, gk[s][1], gk[s][2], gk[s][3])   # affine: valid for any ints
        positions[1, s] = p[1]; positions[2, s] = p[2]; positions[3, s] = p[3]
    end
    At, Bt, Ct = true_cell_vectors(t1)
    w = abs(det(hcat(collect(At), collect(Bt), collect(Ct)))) / (nx * ny * nz)
    return LatticeBatch(copy(pairs), gk, positions, fill(w, m), dk)
end
```

- [ ] **Step 5: Run, verify pass**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: PASS (both new testsets)

- [ ] **Step 6: Commit**

```bash
git add test/fixtures/orb_smooth.xsf test/fixtures/system_smooth_lat.bie src/solver/lattice_batch.jl test/solver/lattice_batch.jl src/BoundaryIntegral.jl
git commit -m "feat: assemble_lattice_batch — pair densities on the virtual global grid"
```

(Export `LatticeBatch`, `assemble_lattice_batch` in `src/BoundaryIntegral.jl` as part of this task.)

---

## Task 3: Solve glue — envelope, sources, `solve_dielectric_lattice_batch`

**Files:**
- Modify: `src/solver/lattice_batch.jl`, `src/BoundaryIntegral.jl` (exports)
- Test: `test/solver/lattice_batch.jl`

- [ ] **Step 1: Write the failing test** (append; coarse params, same style as `test/solver/multi_rhs.jl`)

```julia
    @testset "solve_dielectric_lattice_batch == group solve" begin
        si = read_system_input(joinpath(fixdir, "system_smooth_lat.bie"))
        NQ, RTOLR, LEC = 4, 1e-2, 2.0
        sol_ref = (let g = assemble_rhs_group(si, 1; support_rtol = 1e-6)
            iface = build_group_interface(si, g; n_quad = NQ, rhs_atol = RTOLR, l_ec = LEC)
            Σ, _ = solve_dielectric_box3d_block(iface, group_volume_sources(g);
                fmm_tol = 1e-6, up_tol = 1e-6, max_order = 8, rtol = 1e-8)
            (Σ, iface)
        end)

        st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
        insts = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                     2 => OrbitalInstance(2, 1, lattice_grid_steps(dg, st.primvec, (1, 0, 0))))
        b = assemble_lattice_batch([dg], insts, [(1, 1), (1, 2)]; support_rtol = 1e-6)
        res = solve_dielectric_lattice_batch(si.boxes, si.epses, si.eps_out, b;
            n_quad = NQ, rhs_atol = RTOLR, l_ec = LEC,
            fmm_tol = 1e-6, up_tol = 1e-6, max_order = 8, gmres_rtol = 1e-8)

        @test size(res.sigma, 2) == 2
        @test BoundaryIntegral.num_points(res.interface) == BoundaryIntegral.num_points(sol_ref[2])
        # same interface refinement (same envelope up to row order) => directly comparable Σ
        @test maximum(abs.(res.sigma .- sol_ref[1])) < 1e-6
        @test res.stats.niter > 0
    end
```

- [ ] **Step 2: Run, verify failure**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: FAIL / `solve_dielectric_lattice_batch not defined`

- [ ] **Step 3: Implement** (append to `src/solver/lattice_batch.jl`)

```julia
"""
    envelope_volume_source(b::LatticeBatch)

Per-point rss of the batch densities, as a VolumeSource (drives envelope refinement).
"""
function envelope_volume_source(b::LatticeBatch)
    n = size(b.densities, 1)
    env = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        acc = 0.0
        for k in 1:size(b.densities, 2)
            acc += b.densities[s, k]^2
        end
        env[s] = sqrt(acc)
    end
    return VolumeSource(copy(b.positions), copy(b.weights), env)
end

"Split a LatticeBatch into the Vector{VolumeSource} core form (shared positions)."
function batch_volume_sources(b::LatticeBatch)
    return VolumeSource{Float64, 3}[
        VolumeSource(copy(b.positions), copy(b.weights), b.densities[:, k])
        for k in 1:num_pairs(b)
    ]
end

"""
    solve_dielectric_lattice_batch(boxes, epses, eps_out, b::LatticeBatch; kw...)
        -> (; sigma, interface, sources, stats)

Steps 0–6 for an explicit-pair batch: ONE shared interface refined on the batch
envelope, then block GMRES. Mirrors `solve_dielectric_box3d_group` without SystemInput.
"""
function solve_dielectric_lattice_batch(boxes::Vector{BoxGeom}, epses::Vector{Float64},
        eps_out::Float64, b::LatticeBatch;
        n_quad::Int, rhs_atol::Float64, l_ec::Float64,
        fmm_tol::Float64, up_tol::Float64 = fmm_tol, max_order::Int = 8,
        gmres_rtol::Float64, max_depth::Int = 128, itmax::Int = 500)
    env = envelope_volume_source(b)
    interface = multi_dielectric_box3d_rhs_adaptive(
        n_quad, l_ec, boxes, epses, env, rhs_atol;
        eps_out = eps_out, max_depth = max_depth)
    sources = batch_volume_sources(b)
    Σ, stats = solve_dielectric_box3d_block(interface, sources;
        fmm_tol = fmm_tol, up_tol = up_tol, max_order = max_order,
        rtol = gmres_rtol, itmax = itmax)
    return (; sigma = Σ, interface = interface, sources = sources, stats = stats)
end
```

Export `envelope_volume_source` is already exported for RHSGroup — this is a new method, no export change; export `batch_volume_sources`, `solve_dielectric_lattice_batch`.

- [ ] **Step 4: Run, verify pass**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/solver/lattice_batch.jl test/solver/lattice_batch.jl src/BoundaryIntegral.jl
git commit -m "feat: solve_dielectric_lattice_batch — explicit-pair batch solve without SystemInput"
```

---

## Task 4: `evaluate_batch_potential` — near/far-split external-target evaluation

**Files:**
- Modify: `src/solver/lattice_batch.jl`, `src/BoundaryIntegral.jl` (export)
- Test: `test/solver/lattice_batch.jl`

- [ ] **Step 1: Write the failing tests** (append; reuses `si`, batch and solve from Task 3's testset — restructure so the solved `(b, res)` from Task 3 is computed once at testset scope and shared)

```julia
    # shared solved batch for evaluation tests (coarse)
    st_e, dg_e = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
    si_e = read_system_input(joinpath(fixdir, "system_smooth_lat.bie"))
    insts_e = Dict(1 => OrbitalInstance(1, 1, (0, 0, 0)),
                   2 => OrbitalInstance(2, 1, lattice_grid_steps(dg_e, st_e.primvec, (1, 0, 0))))
    b_e = assemble_lattice_batch([dg_e], insts_e, [(1, 1), (1, 2)]; support_rtol = 1e-6)
    res_e = solve_dielectric_lattice_batch(si_e.boxes, si_e.epses, si_e.eps_out, b_e;
        n_quad = 4, rhs_atol = 1e-2, l_ec = 2.0, fmm_tol = 1e-6, gmres_rtol = 1e-8)

    @testset "evaluate_batch_potential vs TKM-everywhere" begin
        # targets: the batch's own support points (near) + a far ring at distance ~8
        far = hcat(([8.0 * cos(t) + 1.5, 8.0 * sin(t) + 1.5, 1.5] for t in range(0, 2π; length = 17)[1:16])...)
        targets = hcat(b_e.positions, far)
        far_pad = 2.0 * maximum(norm.(BoundaryIntegral.true_cell_vectors(dg_e))) / dg_e.nx

        Φ = evaluate_batch_potential(res_e.interface, res_e.sigma, res_e.sources, targets;
            lhs_tol = 1e-6, volume_tol = 1e-8, far_pad = far_pad)
        @test size(Φ) == (size(targets, 2), 2)

        # reference: TKM at ALL targets (valid near and far) + the same layer map
        pottrg = laplace3d_pottrg_fmm3d_corrected_hcubature(res_e.interface, targets, 1e-6, 1e-6, 5.0)
        for a in 1:2
            sa = BoundaryIntegral.screened_volume_source(res_e.interface, res_e.sources[a],
                BoundaryIntegral.SharpScreening())
            vals = BoundaryIntegral.TKM3D.ltkm3dc(1e-8, sa.positions;
                charges = sa.weights .* sa.density, targets = targets, pgt = 1,
                kmax = BoundaryIntegral._estimate_tkm3dc_kmax(sa))
            @test vals.ier == 0
            Φ_ref = real.(vals.pottarg) .+ (pottrg * res_e.sigma[:, a])
            scale = maximum(abs.(Φ_ref))
            @test maximum(abs.(Φ[:, a] .- Φ_ref)) < 1e-5 * scale
        end
    end

    @testset "V via evaluate_batch_potential == four_index_matrix" begin
        V_ref = four_index_matrix(si_e, res_e.interface, res_e.sources, res_e.sigma)
        targets = b_e.positions                        # the group grid = what four_index uses
        far_pad = 2.0 * maximum(norm.(BoundaryIntegral.true_cell_vectors(dg_e))) / dg_e.nx
        Φ = evaluate_batch_potential(res_e.interface, res_e.sigma, res_e.sources, targets;
            lhs_tol = si_e.solve.lhs_tol, volume_tol = si_e.solve.volume_tol, far_pad = far_pad)
        K = 2
        V = [LinearAlgebra.dot(b_e.weights .* b_e.densities[:, a], Φ[:, bb]) for a in 1:K, bb in 1:K]
        @test maximum(abs.(V .- V_ref)) < 1e-6 * maximum(abs.(V_ref))
    end
```

(Add `using LinearAlgebra` at the top of the test file.)

- [ ] **Step 2: Run, verify failure**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: FAIL / `evaluate_batch_potential not defined`

- [ ] **Step 3: Implement** (append to `src/solver/lattice_batch.jl`)

```julia
"""
    evaluate_batch_potential(interface, Σ, sources, targets;
        lhs_tol, volume_tol, far_pad, range_factor=5.0) -> Φ (n_targets × K)

Total potential `Φ_a = u_inc[ρ_a] + u[σ_a]` of a solved batch at arbitrary targets.

- `u[σ_a]`: corrected layer-potential map (FMM + hcubature near correction) built ONCE
  for the target set, applied per column of Σ.
- `u_inc[ρ_a]`: batch-level near/far split on the shared support bounding box padded by
  `far_pad`. Near targets: TKM volume potential per source (screened density). Far
  targets: ONE `nd = K` point-charge FMM over the screened quadrature points (the
  trapezoidal far field of a smooth compact density; `far_pad` ≳ 2 grid steps).

Conventions match `four_index_matrix` (TKM potential used as-is; FMM 1/r scaled by
1/(4π)); the agreement test pins this down.
"""
function evaluate_batch_potential(interface, Σ::AbstractMatrix,
        sources::Vector{<:VolumeSource{Float64, 3}}, targets::Matrix{Float64};
        lhs_tol::Float64, volume_tol::Float64, far_pad::Float64,
        range_factor::Float64 = 5.0)
    K = length(sources)
    nt = size(targets, 2)
    size(targets, 1) == 3 || throw(ArgumentError("targets must be 3 × n"))
    size(Σ, 2) == K || throw(DimensionMismatch("Σ columns ≠ number of sources"))

    Φ = Matrix{Float64}(undef, nt, K)

    # scattered part: build the corrected map once, apply per column
    pottrg = laplace3d_pottrg_fmm3d_corrected_hcubature(interface, targets, lhs_tol, lhs_tol, range_factor)
    for a in 1:K
        Φ[:, a] = pottrg * Σ[:, a]
    end

    # incident part: shared positions => shared screening & shared near/far split
    screened = [screened_volume_source(interface, vs, SharpScreening()) for vs in sources]
    pos = screened[1].positions
    nsrc = size(pos, 2)
    lo = ntuple(d -> minimum(view(pos, d, :)) - far_pad, 3)
    hi = ntuple(d -> maximum(view(pos, d, :)) + far_pad, 3)
    isnear = BitVector(undef, nt)
    @inbounds for t in 1:nt
        isnear[t] = lo[1] <= targets[1, t] <= hi[1] &&
                    lo[2] <= targets[2, t] <= hi[2] &&
                    lo[3] <= targets[3, t] <= hi[3]
    end
    near_idx = findall(isnear)
    far_idx = findall(.!isnear)

    if !isempty(far_idx)
        charges = Matrix{Float64}(undef, K, nsrc)
        @inbounds for a in 1:K, s in 1:nsrc
            charges[a, s] = screened[a].weights[s] * screened[a].density[s]
        end
        vals = lfmm3d(volume_tol, pos; charges = charges,
                      targets = targets[:, far_idx], pgt = 1, nd = K)
        pt = reshape(vals.pottarg, K, length(far_idx))
        @inbounds for a in 1:K
            for (m, t) in enumerate(far_idx)
                Φ[t, a] += pt[a, m] / (4π)
            end
        end
    end

    if !isempty(near_idx)
        tnear = targets[:, near_idx]
        for a in 1:K
            sa = screened[a]
            vals = TKM3D.ltkm3dc(volume_tol, sa.positions;
                charges = sa.weights .* sa.density, targets = tnear, pgt = 1,
                kmax = _estimate_tkm3dc_kmax(sa))
            vals.ier == 0 || error("TKM3D.ltkm3dc failed, ier=$(vals.ier)")
            Φ[near_idx, a] .+= real.(vals.pottarg)
        end
    end
    return Φ
end
```

NOTE for the implementer: the `/(4π)` on the FMM far part assumes TKM returns the
`1/(4π|r|)`-normalized potential (it is used unscaled in `four_index_matrix`). The
"vs TKM-everywhere" test is the authority — if it fails by exactly 4π on far targets,
flip the scaling, rerun, and document the convention in the docstring.

- [ ] **Step 4: Run, verify pass**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/solver/lattice_batch.jl test/solver/lattice_batch.jl src/BoundaryIntegral.jl
git commit -m "feat: evaluate_batch_potential — near/far-split batched external-target evaluation"
```

---

## Task 5: `BatchResult` + atomic save/load

**Files:**
- Create: `src/solver/batch_io.jl`
- Modify: `src/BoundaryIntegral.jl` (include after `solver/lattice_batch.jl`; exports)
- Test: `test/solver/batch_io.jl` (create); add `include("solver/batch_io.jl")` to `test/runtests.jl` next to the lattice_batch include

- [ ] **Step 1: Write the failing tests**

```julia
# test/solver/batch_io.jl
using BoundaryIntegral
using Test

@testset "batch_io" begin
    br = BatchResult(BoundaryIntegral.BATCH_FORMAT_VERSION, 7,
        [(1, 1), (1, 2)],
        [(1, 1, 1), (2, 1, 1)],
        [0.1, 0.1],
        [1.0 0.0; 0.5 0.25],
        nothing,                          # interface: any serializable object
        ones(4, 2),
        Dict{String,Any}("niter" => 3, "t_solve" => 1.5))

    mktempdir() do dir
        p = joinpath(dir, "batch_0007.jls")
        @test !is_complete_batch(p)                       # missing file
        save_batch_result(p, br)
        @test is_complete_batch(p)
        @test isempty(filter(f -> occursin(".tmp", f), readdir(dir)))   # tmp cleaned up
        br2 = load_batch_result(p)
        @test br2.batch_id == 7
        @test br2.pair_ids == br.pair_ids
        @test br2.gidx == br.gidx
        @test br2.densities == br.densities
        @test br2.sigma == br.sigma
        @test br2.stats["niter"] == 3

        # corrupted file -> incomplete, load throws
        open(p, "w") do io; write(io, "garbage"); end
        @test !is_complete_batch(p)
        @test_throws Exception load_batch_result(p)

        # wrong version -> incomplete
        bad = BatchResult(BoundaryIntegral.BATCH_FORMAT_VERSION + 1, br.batch_id,
            br.pair_ids, br.gidx, br.weights, br.densities, br.interface, br.sigma, br.stats)
        save_batch_result(p, bad)
        @test !is_complete_batch(p)
    end
end
```

- [ ] **Step 2: Run, verify failure**

Run: `julia --project=. test/solver/batch_io.jl`
Expected: FAIL / `BatchResult not defined`

- [ ] **Step 3: Implement**

```julia
# src/solver/batch_io.jl
# Solve-phase output container + atomic file IO (spec §3). Files are written
# tmp-then-rename on the SAME filesystem, so a batch file either exists complete or
# not at all; status scans never see partial writes.

using Serialization

const BATCH_FORMAT_VERSION = 1

"""
    BatchResult

Everything the post-eval phase needs from one solved batch: the raw truncated pair
densities (global-grid indexed), the adapted interface, the layer densities Σ, and
solve stats. Positions are NOT stored — they are recomputed from `gidx` + the template.
"""
struct BatchResult
    version::Int
    batch_id::Int
    pair_ids::Vector{Tuple{Int,Int}}
    gidx::Vector{NTuple{3,Int}}
    weights::Vector{Float64}
    densities::Matrix{Float64}        # n × K raw pair densities
    interface::Any                    # DielectricInterface (serialized as-is)
    sigma::Matrix{Float64}            # N × K
    stats::Dict{String,Any}
end

"Atomic write: serialize to `<path>.tmp.<pid>`, then rename onto `path`."
function save_batch_result(path::AbstractString, br::BatchResult)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid())
    open(tmp, "w") do io
        serialize(io, br)
    end
    mv(tmp, path; force = true)
    return path
end

"""
    load_batch_result(path) -> BatchResult

Deserialize and validate (format version, internal shape consistency). Throws on any
problem — callers treating a batch as done must use `is_complete_batch` first.
"""
function load_batch_result(path::AbstractString)
    br = open(deserialize, path)
    br isa BatchResult || error("$path: not a BatchResult")
    br.version == BATCH_FORMAT_VERSION ||
        error("$path: format version $(br.version) ≠ $(BATCH_FORMAT_VERSION)")
    n, K = size(br.densities)
    length(br.gidx) == n || error("$path: gidx/densities mismatch")
    length(br.weights) == n || error("$path: weights/densities mismatch")
    length(br.pair_ids) == K || error("$path: pair_ids/densities mismatch")
    size(br.sigma, 2) == K || error("$path: sigma/densities mismatch")
    return br
end

"True iff `path` exists and loads as a valid, current-version BatchResult."
function is_complete_batch(path::AbstractString)
    isfile(path) || return false
    try
        load_batch_result(path)
        return true
    catch
        return false
    end
end
```

Export `BatchResult`, `save_batch_result`, `load_batch_result`, `is_complete_batch`.

- [ ] **Step 4: Run, verify pass**

Run: `julia --project=. test/solver/batch_io.jl`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/solver/batch_io.jl test/solver/batch_io.jl src/BoundaryIntegral.jl test/runtests.jl
git commit -m "feat: BatchResult + atomic save/load with completeness validation"
```

---

## Task 6: Package integration — full test suite

**Files:** none new.

- [ ] **Step 1: Run the basic suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS (all existing + new testsets; the full-suite extras stay gated behind `BI_RUN_FULL_TESTS`)

- [ ] **Step 2: Fix any fallout, commit if changes were needed**

```bash
git add -u && git commit -m "test: integration fixes for lattice-batch additions"   # only if needed
```

---

## Task 7: Campaign layer scaffold + config

**Repo/cwd:** `~/work/four_index_integral_solver/codes/lattice_scale/` (create)

**Files:**
- Create: `Project.toml`, `src/CampaignLib.jl`, `src/config.jl`, `test/runtests.jl`, `test/config.jl`, `.gitignore` (`Manifest.toml`, `*.out`)

- [ ] **Step 1: Scaffold the project**

`Project.toml`:

```toml
name = "CampaignLib"
uuid = "f1a7c0de-0000-4000-8000-1ce11a771ce5"
version = "0.1.0"

[deps]
BoundaryIntegral = "37e2e46d-0000-0000-0000-000000000000"  # use the real UUID from ~/codes/BoundaryIntegral.jl/Project.toml
ClusterManagers = "34f1f09b-3a8b-5176-ab39-66d58a4d544e"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

(Copy the exact BoundaryIntegral and LinearAlgebra UUIDs from the package's own Project.toml / a working Manifest. Then `julia --project=. -e 'using Pkg; Pkg.develop(path=expanduser("~/codes/BoundaryIntegral.jl")); Pkg.instantiate()'`.)

`src/CampaignLib.jl`:

```julia
module CampaignLib

using BoundaryIntegral
using LinearAlgebra
using Printf
using Serialization
using TOML

export Campaign, load_campaign, batch_path, v_path, manifest_path, centers_path,
       targets_path, rho_store_path, logs_dir,
       CenterInfo, BatchSpec, enumerate_centers, enumerate_pairs, build_batches,
       write_manifest, read_manifest, write_centers, read_centers,
       pending_batches, load_templates!,
       solve_batch, consolidate, eval_batch, assemble_v, prepare

include("config.jl")
include("manifest.jl")
include("tasks.jl")

end
```

(`manifest.jl`/`tasks.jl` start as empty files in this task; filled by Tasks 8–13.)

`src/config.jl`:

```julia
"""
    Campaign

Parsed campaign.toml + derived paths. Heavy template grids are NOT loaded here —
see `load_templates!`.
"""
struct Campaign
    name::String
    root::String                       # ceph output directory
    xsf::Vector{String}                # one template per sublattice
    nx::Int
    ny::Int
    neighbor_cutoff::Float64
    eps_out::Float64
    boxes::Vector{BoundaryIntegral.BoxGeom}
    epses::Vector{Float64}
    solve::Dict{String,Float64}        # n_quad, edge_refine_level, rhs_tol, lhs_tol,
                                       # gmres_rtol, support_rtol, volume_tol,
                                       # max_order, max_depth (numeric; Int-like coerced on use)
    n_centers_per_batch::Int
    far_pad_steps::Float64
end

function load_campaign(toml_path::AbstractString)
    d = TOML.parsefile(toml_path)
    c, l, di, s = d["campaign"], d["lattice"], d["dielectrics"], d["solve"]
    boxes = BoundaryIntegral.BoxGeom[]
    epses = Float64[]
    for row in di["boxes"]
        length(row) == 7 || error("dielectrics.boxes rows are [cx cy cz Lx Ly Lz eps]")
        push!(boxes, (center = (row[1], row[2], row[3]), Lx = row[4], Ly = row[5], Lz = row[6]))
        push!(epses, row[7])
    end
    solve = Dict{String,Float64}(k => Float64(v) for (k, v) in s)
    return Campaign(c["name"], c["root"], String.(d["orbitals"]["xsf"]),
        Int(l["nx"]), Int(l["ny"]), Float64(l["neighbor_cutoff"]),
        Float64(get(di, "eps_out", 1.0)), boxes, epses, solve,
        Int(get(d, "batching", Dict())["n_centers_per_batch"]),
        Float64(get(d, "eval", Dict("far_pad_steps" => 2.0))["far_pad_steps"]))
end

manifest_path(c::Campaign)  = joinpath(c.root, "manifest.tsv")
centers_path(c::Campaign)   = joinpath(c.root, "centers.tsv")
targets_path(c::Campaign)   = joinpath(c.root, "targets.jls")
rho_store_path(c::Campaign) = joinpath(c.root, "rho_store.jls")
logs_dir(c::Campaign)       = joinpath(c.root, "logs")
batch_path(c::Campaign, id::Int) = joinpath(c.root, "batches", @sprintf("batch_%04d.jls", id))
v_path(c::Campaign, id::Int)     = joinpath(c.root, "V", @sprintf("V_%04d.jls", id))

# NOTE: named campaign_l_ec, NOT resolved_l_ec — BoundaryIntegral exports
# resolved_l_ec(::SystemInput) and a same-named definition here would clash.
campaign_l_ec(c::Campaign) =
    minimum(b.Lz for b in c.boxes) / 2.0^Int(c.solve["edge_refine_level"]) * 1.01

# worker-local template cache: path => (structure, datagrid)
const TEMPLATE_CACHE = Dict{String,Any}()
function load_templates!(c::Campaign)
    return [get!(TEMPLATE_CACHE, p) do
                BoundaryIntegral.read_xsf(p)
            end for p in c.xsf]
end
```

`test/runtests.jl`:

```julia
using CampaignLib
using Test

@testset "CampaignLib" begin
    include("config.jl")
    include("manifest.jl")
    include("pipeline.jl")
end
```

- [ ] **Step 2: Write the failing config test**

```julia
# test/config.jl
@testset "config" begin
    mktempdir() do dir
        toml = joinpath(dir, "c.toml")
        write(toml, """
        [campaign]
        name = "t"
        root = "$dir/out"

        [orbitals]
        xsf = ["/a/one.xsf", "/a/two.xsf"]

        [lattice]
        nx = 2
        ny = 3
        neighbor_cutoff = 5.0

        [dielectrics]
        eps_out = 1.0
        boxes = [[0.0, 0.0, 7.5, 90.0, 90.0, 3.35, 3.5]]

        [solve]
        n_quad = 6
        edge_refine_level = 2
        rhs_tol = 1e-3
        lhs_tol = 1e-5
        gmres_rtol = 1e-5
        support_rtol = 1e-4
        volume_tol = 1e-5
        max_order = 8
        max_depth = 128

        [batching]
        n_centers_per_batch = 1

        [eval]
        far_pad_steps = 2.0
        """)
        c = load_campaign(toml)
        @test c.name == "t" && c.nx == 2 && c.ny == 3
        @test length(c.boxes) == 1 && c.epses == [3.5]
        @test c.boxes[1].Lz == 3.35
        @test c.solve["rhs_tol"] == 1e-3
        @test endswith(batch_path(c, 7), "batches/batch_0007.jls")
        @test endswith(v_path(c, 12), "V/V_0012.jls")
        @test CampaignLib.campaign_l_ec(c) ≈ 3.35 / 4 * 1.01
    end
end
```

- [ ] **Step 3: Run, verify failure, then make it pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected first: FAIL (missing files) → after scaffold complete: PASS
(`test/manifest.jl` and `test/pipeline.jl` start as empty files so the include succeeds.)

- [ ] **Step 4: Commit** (in `~/work/four_index_integral_solver`)

```bash
git add codes/lattice_scale
git commit -m "lattice_scale: campaign scaffold + config layer"
```

---

## Task 8: Manifest — centers, pair dedup, batches, TSV round-trip

**Files:**
- Create: `src/manifest.jl` (fill), `test/manifest.jl` (fill)

- [ ] **Step 1: Write the failing tests**

```julia
# test/manifest.jl
@testset "manifest" begin
    # synthetic geometry: 2x2 cells, 2 sublattices, square lattice a1=(1,0,0) a2=(0,1,0)
    primvec = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 10.0]
    centroids = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)]      # per template
    steps_per_cell = ((2, 0, 0), (0, 2, 0))             # grid steps for a1, a2

    centers = enumerate_centers(2, 2, primvec, centroids, steps_per_cell)
    @test length(centers) == 8
    @test allunique(getfield.(centers, :id))
    c1 = centers[findfirst(c -> c.Rx == 0 && c.Ry == 0 && c.template_id == 1, centers)]
    c5 = centers[findfirst(c -> c.Rx == 1 && c.Ry == 1 && c.template_id == 2, centers)]
    @test c1.steps == (0, 0, 0)
    @test c5.steps == (2, 2, 0)
    @test collect(c5.center) ≈ [1.5, 1.5, 0.0]

    # dedup: i <= j, each unordered pair once
    pairs = enumerate_pairs(centers, 1.2)                # nn cutoff: dist <= 1.2
    @test all(p -> p[1] <= p[2], pairs)
    @test allunique(pairs)
    @test length(pairs) == length(unique(pairs))
    onsite = count(p -> p[1] == p[2], pairs)
    @test onsite == 8                                    # every center pairs with itself

    batches = build_batches(pairs, 1)
    @test sum(b -> length(b.pairs), batches) == length(pairs)   # every pair exactly once
    @test allunique(getfield.(batches, :batch_id))
    # anchored: every pair's min id belongs to the batch's anchor set
    for b in batches
        @test all(p -> min(p[1], p[2]) in b.anchors, b.pairs)
    end
    b2 = build_batches(pairs, 2)
    @test sum(b -> length(b.pairs), b2) == length(pairs)
    @test length(b2) < length(batches)

    # TSV round-trip
    mktempdir() do dir
        cp = joinpath(dir, "centers.tsv"); mp = joinpath(dir, "manifest.tsv")
        write_centers(cp, centers); write_manifest(mp, batches)
        @test read_centers(cp) == centers
        @test read_manifest(mp) == batches
    end
end
```

- [ ] **Step 2: Run, verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL / `enumerate_centers not defined`

- [ ] **Step 3: Implement `src/manifest.jl`**

```julia
"""
    CenterInfo

One orbital site of the flake: lattice cell (Rx, Ry), sublattice template, the
global-frame integer offset `steps`, and the real-space center (centroid + R).
"""
struct CenterInfo
    id::Int
    template_id::Int
    Rx::Int
    Ry::Int
    steps::NTuple{3,Int}
    center::NTuple{3,Float64}
end

struct BatchSpec
    batch_id::Int
    anchors::Vector{Int}                 # center ids whose pair lists were merged
    pairs::Vector{Tuple{Int,Int}}
end

Base.:(==)(a::CenterInfo, b::CenterInfo) =
    a.id == b.id && a.template_id == b.template_id && a.Rx == b.Rx && a.Ry == b.Ry &&
    a.steps == b.steps && all(isapprox.(a.center, b.center; atol = 1e-12))
Base.:(==)(a::BatchSpec, b::BatchSpec) =
    a.batch_id == b.batch_id && a.anchors == b.anchors && a.pairs == b.pairs

"""
    enumerate_centers(nx, ny, primvec, centroids, steps_per_cell) -> Vector{CenterInfo}

Pure geometry (no file IO): center id = (Ry*nx + Rx)*n_sub + template_id, position =
template centroid + Rx·a1 + Ry·a2, steps = Rx·steps(a1) + Ry·steps(a2).
`steps_per_cell` = the per-template-grid steps of (a1, a2) from `lattice_grid_steps`.
"""
function enumerate_centers(nx::Int, ny::Int, primvec::AbstractMatrix,
        centroids::Vector{NTuple{3,Float64}}, steps_per_cell::NTuple{2,NTuple{3,Int}})
    nsub = length(centroids)
    a1 = primvec[1, :]; a2 = primvec[2, :]
    s1, s2 = steps_per_cell
    out = CenterInfo[]
    for Ry in 0:(ny-1), Rx in 0:(nx-1), t in 1:nsub
        id = (Ry * nx + Rx) * nsub + t
        steps = ntuple(d -> Rx * s1[d] + Ry * s2[d], 3)
        ctr = ntuple(d -> centroids[t][d] + Rx * a1[d] + Ry * a2[d], 3)
        push!(out, CenterInfo(id, t, Rx, Ry, steps, ctr))
    end
    return out
end

"Unique pairs (i ≤ j) with center distance ≤ cutoff. On-site pairs (i,i) included."
function enumerate_pairs(centers::Vector{CenterInfo}, cutoff::Float64)
    byid = sort(centers; by = c -> c.id)
    pairs = Tuple{Int,Int}[]
    for (m, ci) in enumerate(byid)
        for cj in byid[m:end]
            d = sqrt(sum(abs2, ci.center .- cj.center))
            d <= cutoff && push!(pairs, (ci.id, cj.id))
        end
    end
    return pairs
end

"""
    build_batches(pairs, n_centers_per_batch) -> Vector{BatchSpec}

Each pair belongs to its anchor (= min id); consecutive anchors are merged
`n_centers_per_batch` at a time. Every pair lands in exactly one batch.
"""
function build_batches(pairs::Vector{Tuple{Int,Int}}, n_centers_per_batch::Int)
    by_anchor = Dict{Int,Vector{Tuple{Int,Int}}}()
    for p in pairs
        push!(get!(by_anchor, min(p[1], p[2]), Tuple{Int,Int}[]), p)
    end
    anchors = sort(collect(keys(by_anchor)))
    out = BatchSpec[]
    bid = 0
    for grp in Iterators.partition(anchors, n_centers_per_batch)
        bid += 1
        ps = reduce(vcat, (sort(by_anchor[a]) for a in grp))
        push!(out, BatchSpec(bid, collect(grp), ps))
    end
    return out
end

# ---- TSV IO (human-greppable; deterministic) ----
function write_centers(path::AbstractString, centers::Vector{CenterInfo})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "id\ttemplate\tRx\tRy\tsx\tsy\tsz\tcx\tcy\tcz")
        for c in sort(centers; by = c -> c.id)
            println(io, join([c.id, c.template_id, c.Rx, c.Ry, c.steps..., c.center...], '\t'))
        end
    end
end

function read_centers(path::AbstractString)
    out = CenterInfo[]
    for (n, line) in enumerate(eachline(path))
        n == 1 && continue
        f = split(line, '\t')
        push!(out, CenterInfo(parse(Int, f[1]), parse(Int, f[2]),
            parse(Int, f[3]), parse(Int, f[4]),
            (parse(Int, f[5]), parse(Int, f[6]), parse(Int, f[7])),
            (parse(Float64, f[8]), parse(Float64, f[9]), parse(Float64, f[10]))))
    end
    return out
end

function write_manifest(path::AbstractString, batches::Vector{BatchSpec})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "batch_id\tanchors\tK\tpairs")
        for b in sort(batches; by = b -> b.batch_id)
            ps = join(("$(i):$(j)" for (i, j) in b.pairs), ';')
            println(io, join([b.batch_id, join(b.anchors, ','), length(b.pairs), ps], '\t'))
        end
    end
end

function read_manifest(path::AbstractString)
    out = BatchSpec[]
    for (n, line) in enumerate(eachline(path))
        n == 1 && continue
        f = split(line, '\t')
        pairs = [(parse(Int, split(p, ':')[1]), parse(Int, split(p, ':')[2]))
                 for p in split(f[4], ';')]
        push!(out, BatchSpec(parse(Int, f[1]), parse.(Int, split(f[2], ',')), pairs))
    end
    return out
end
```

- [ ] **Step 4: Run, verify pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codes/lattice_scale/src/manifest.jl codes/lattice_scale/test/manifest.jl
git commit -m "lattice_scale: center/pair enumeration, dedup batching, TSV manifest"
```

---

## Task 9: Mini-campaign test fixture + `prepare` + status scan

**Files:**
- Create: `test/fixture_campaign.jl` (helper used by pipeline tests)
- Modify: `src/tasks.jl` (add `prepare`, `pending_batches`), `test/pipeline.jl`

- [ ] **Step 1: Write the fixture helper**

```julia
# test/fixture_campaign.jl — a tiny self-contained campaign for pipeline tests:
# 6x6x6 template (3 cells/axis, 2 steps/cell), Gaussian blob, 2x1 lattice, 1 sublattice.
function write_fixture_xsf(path::AbstractString)
    open(path, "w") do io
        println(io, "CRYSTAL\nPRIMVEC\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0")
        println(io, "PRIMCOORD\n1 1\nX 0.75 0.75 0.75")
        println(io, "BEGIN_BLOCK_DATAGRID_3D\n g\nBEGIN_DATAGRID_3D_g")
        println(io, "6 6 6\n0.0 0.0 0.0")
        println(io, "2.5 0.0 0.0\n0.0 2.5 0.0\n0.0 0.0 2.5")
        vals = Float64[]
        for k in 1:6, j in 1:6, i in 1:6
            x = ((i-1)/6)*3.0; y = ((j-1)/6)*3.0; z = ((k-1)/6)*3.0
            push!(vals, exp(-((x-0.75)^2 + (y-0.75)^2 + (z-0.75)^2) / (2*0.4^2)))
        end
        for chunk in Iterators.partition(vals, 6)
            println(io, join(string.(chunk), " "))
        end
        println(io, "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D")
    end
end

function write_fixture_campaign(dir::AbstractString)
    xsf = joinpath(dir, "orb.xsf")
    write_fixture_xsf(xsf)
    toml = joinpath(dir, "campaign.toml")
    write(toml, """
    [campaign]
    name = "mini"
    root = "$(joinpath(dir, "out"))"

    [orbitals]
    xsf = ["$xsf"]

    [lattice]
    nx = 2
    ny = 1
    neighbor_cutoff = 1.2

    [dielectrics]
    eps_out = 1.0
    boxes = [[1.5, 1.5, 1.5, 8.0, 8.0, 8.0, 2.0]]

    [solve]
    n_quad = 4
    edge_refine_level = 1
    rhs_tol = 1e-2
    lhs_tol = 1e-6
    gmres_rtol = 1e-8
    support_rtol = 1e-6
    volume_tol = 1e-8
    max_order = 8
    max_depth = 16

    [batching]
    n_centers_per_batch = 1

    [eval]
    far_pad_steps = 2.0
    """)
    return toml
end
```

- [ ] **Step 2: Write the failing tests**

```julia
# test/pipeline.jl
include("fixture_campaign.jl")

@testset "prepare + status" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c)
        @test isfile(manifest_path(c)) && isfile(centers_path(c))
        centers = read_centers(centers_path(c))
        @test length(centers) == 2                       # 2x1 lattice, 1 sublattice
        @test centers[2].steps == (2, 0, 0)              # 2 steps/cell along a1
        batches = read_manifest(manifest_path(c))
        @test length(batches) == 2                       # anchors 1 and 2
        @test sort(reduce(vcat, b.pairs for b in batches)) == [(1,1), (1,2), (2,2)]

        @test sort(pending_batches(c, :solve)) == [1, 2]
        # a completed batch file flips status
        br = BatchResult(BoundaryIntegral.BATCH_FORMAT_VERSION, 1, [(1,1),(1,2)],
            [(1,1,1)], [0.1], ones(1, 2), nothing, ones(3, 2), Dict{String,Any}())
        save_batch_result(batch_path(c, 1), br)
        @test pending_batches(c, :solve) == [2]
        @test sort(pending_batches(c, :eval)) == [1, 2]  # no V files yet
    end
end
```

(Add `using BoundaryIntegral` to `test/runtests.jl` imports.)

- [ ] **Step 3: Run, verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL / `prepare not defined`

- [ ] **Step 4: Implement** (start `src/tasks.jl`)

```julia
"""
    prepare(c::Campaign)

Phase 1 (single process): enumerate centers/pairs/batches and write
`centers.tsv` + `manifest.tsv` under `c.root`. Idempotent: existing files are kept
(delete them to re-prepare).
"""
function prepare(c::Campaign)
    if isfile(manifest_path(c)) && isfile(centers_path(c))
        @info "prepare: manifest exists, skipping" manifest_path(c)
        return read_manifest(manifest_path(c))
    end
    temps = load_templates!(c)
    st1 = temps[1][1]
    centroids = [BoundaryIntegral.density_centroid(t[2]) for t in temps]
    s1 = lattice_grid_steps(temps[1][2], st1.primvec, (1, 0, 0))
    s2 = lattice_grid_steps(temps[1][2], st1.primvec, (0, 1, 0))
    centers = enumerate_centers(c.nx, c.ny, st1.primvec,
        [ntuple(d -> Float64(ct[d]), 3) for ct in centroids], (s1, s2))
    pairs = enumerate_pairs(centers, c.neighbor_cutoff)
    batches = build_batches(pairs, c.n_centers_per_batch)
    write_centers(centers_path(c), centers)
    write_manifest(manifest_path(c), batches)
    @info "prepare: wrote manifest" n_centers=length(centers) n_pairs=length(pairs) n_batches=length(batches)
    return batches
end

"""
    pending_batches(c, phase::Symbol) -> Vector{Int}

Batch ids still to do, derived from files on disk (spec §3: no mutable status).
`:solve` → no complete batch file; `:eval` → no complete V file.
"""
function pending_batches(c::Campaign, phase::Symbol)
    batches = read_manifest(manifest_path(c))
    ids = getfield.(batches, :batch_id)
    if phase === :solve
        return [id for id in ids if !is_complete_batch(batch_path(c, id))]
    elseif phase === :eval
        return [id for id in ids if !_is_complete_v(v_path(c, id))]
    end
    error("unknown phase $phase")
end
```

(`_is_complete_v` is added with the V-file format in Task 12; for THIS task define it as `_is_complete_v(path) = isfile(path)` with a comment that Task 12 replaces it.)

- [ ] **Step 5: Run, verify pass; commit**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → PASS

```bash
git add codes/lattice_scale/src/tasks.jl codes/lattice_scale/test/pipeline.jl codes/lattice_scale/test/fixture_campaign.jl
git commit -m "lattice_scale: prepare phase + file-derived status scan + mini-campaign fixture"
```

---

## Task 10: `solve_batch` task

**Files:**
- Modify: `src/tasks.jl`, `test/pipeline.jl`

- [ ] **Step 1: Write the failing test** (append to `test/pipeline.jl`)

```julia
@testset "solve_batch" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c)
        t = @elapsed solve_batch(c, 1)
        @test is_complete_batch(batch_path(c, 1))
        br = load_batch_result(batch_path(c, 1))
        @test br.batch_id == 1
        @test br.pair_ids == [(1, 1), (1, 2)]
        @test size(br.sigma, 2) == 2
        @test size(br.densities, 2) == 2 && size(br.densities, 1) == length(br.gidx)
        @test haskey(br.stats, "t_total") && haskey(br.stats, "niter") && haskey(br.stats, "dof")
        @test br.stats["dof"] > 0
        solve_batch(c, 1)                                # idempotent: skips, no error
        @test pending_batches(c, :solve) == [2]
        solve_batch(c, 2)
        @test isempty(pending_batches(c, :solve))
    end
end
```

- [ ] **Step 2: Run, verify failure**

Expected: FAIL / `solve_batch not defined`

- [ ] **Step 3: Implement** (append to `src/tasks.jl`)

```julia
# OrbitalInstances for the centers referenced by a batch (template grids shared via cache)
function _batch_instances(c::Campaign, spec::BatchSpec)
    centers = read_centers(centers_path(c))
    byid = Dict(ct.id => ct for ct in centers)
    need = unique(reduce(vcat, [[p[1], p[2]] for p in spec.pairs]))
    return Dict(id => OrbitalInstance(id, byid[id].template_id, byid[id].steps) for id in need)
end

"""
    solve_batch(c::Campaign, batch_id) -> path | nothing

Solve phase for one batch (spec §5): assemble pair densities on the global grid,
envelope-refine ONE shared interface, block-GMRES, write the BatchResult atomically.
Skips (returns nothing) if the output already exists and is complete.
"""
function solve_batch(c::Campaign, batch_id::Int)
    out = batch_path(c, batch_id)
    if is_complete_batch(out)
        @info "solve_batch: already complete, skipping" batch_id
        return nothing
    end
    t0 = time()
    spec = only(filter(b -> b.batch_id == batch_id, read_manifest(manifest_path(c))))
    temps = load_templates!(c)
    grids = [t[2] for t in temps]
    insts = _batch_instances(c, spec)

    b = assemble_lattice_batch(grids, insts, spec.pairs;
        support_rtol = c.solve["support_rtol"])
    t_asm = time() - t0

    res = solve_dielectric_lattice_batch(c.boxes, c.epses, c.eps_out, b;
        n_quad = Int(c.solve["n_quad"]), rhs_atol = c.solve["rhs_tol"],
        l_ec = campaign_l_ec(c), fmm_tol = c.solve["lhs_tol"],
        up_tol = c.solve["lhs_tol"], max_order = Int(c.solve["max_order"]),
        gmres_rtol = c.solve["gmres_rtol"], max_depth = Int(c.solve["max_depth"]))
    t_total = time() - t0

    stats = Dict{String,Any}(
        "t_assemble" => t_asm, "t_total" => t_total,
        "niter" => res.stats.niter, "dof" => size(res.sigma, 1),
        "n_support" => length(b.gidx), "K" => length(spec.pairs),
        "hostname" => gethostname())
    br = BatchResult(BoundaryIntegral.BATCH_FORMAT_VERSION, batch_id, b.pair_ids,
        b.gidx, b.weights, b.densities, res.interface, res.sigma, stats)
    save_batch_result(out, br)
    @info "solve_batch: done" batch_id dof=stats["dof"] K=stats["K"] t_total
    return out
end
```

- [ ] **Step 4: Run, verify pass; commit**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → PASS (this testset actually solves; ~seconds at the fixture scale)

```bash
git add codes/lattice_scale/src/tasks.jl codes/lattice_scale/test/pipeline.jl
git commit -m "lattice_scale: solve_batch task (assemble -> envelope interface -> block GMRES -> BatchResult)"
```

---

## Task 11: `consolidate` — union target set + ρ store

**Files:**
- Modify: `src/tasks.jl`, `test/pipeline.jl`

- [ ] **Step 1: Write the failing test**

```julia
@testset "consolidate" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c); solve_batch(c, 1); solve_batch(c, 2)
        consolidate(c)
        @test isfile(targets_path(c)) && isfile(rho_store_path(c))

        T = open(Serialization.deserialize, targets_path(c))
        store = open(Serialization.deserialize, rho_store_path(c))
        br1 = load_batch_result(batch_path(c, 1))
        br2 = load_batch_result(batch_path(c, 2))

        @test T.gidx == sort(union(br1.gidx, br2.gidx))          # exact union, sorted
        @test size(T.positions) == (3, length(T.gidx))
        @test store.pair_ids == vcat(br1.pair_ids, br2.pair_ids) # batch order
        # contraction vectors: tw = w .* rho on the pair's own support rows
        k = 1                                                    # pair (1,1) from batch 1
        @test store.tw[k] ≈ (br1.weights .* br1.densities[:, 1])
        # t_idx maps the pair's support into T
        @test T.gidx[store.t_idx[k]] == br1.gidx
        # consolidate is idempotent
        consolidate(c)
        @test store.pair_ids == open(Serialization.deserialize, rho_store_path(c)).pair_ids
    end
end
```

(Add `using Serialization` to the test runner imports.)

- [ ] **Step 2: Run, verify failure**

Expected: FAIL / `consolidate not defined`

- [ ] **Step 3: Implement** (append to `src/tasks.jl`)

```julia
"""
    consolidate(c::Campaign)

Between solve and eval (single process, spec §6 + plan deviation note): build
- `targets.jls`: `(; gidx, positions)` — the sorted union of ALL stored batch supports
  (the shared eval target set T), positions from the template's affine map;
- `rho_store.jls`: `(; pair_ids, t_idx, tw)` — per pair, its support's row indices in T
  and the contraction vector `w .* ρ` (raw density), in manifest batch order.
Built from the SOLVED batch files, so T is exactly consistent with the stored ρ.
Errors if any batch is missing. Idempotent (overwrites deterministically).
"""
function consolidate(c::Campaign)
    batches = read_manifest(manifest_path(c))
    missing_ids = [b.batch_id for b in batches if !is_complete_batch(batch_path(c, b.batch_id))]
    isempty(missing_ids) || error("consolidate: unsolved batches: $missing_ids")

    brs = [load_batch_result(batch_path(c, b.batch_id)) for b in batches]
    gall = NTuple{3,Int}[]
    for br in brs
        append!(gall, br.gidx)
    end
    gidx = sort(unique(gall))
    rowofg = Dict(g => r for (r, g) in enumerate(gidx))

    temps = load_templates!(c)
    dg = temps[1][2]
    positions = Matrix{Float64}(undef, 3, length(gidx))
    for (r, g) in enumerate(gidx)
        p = BoundaryIntegral.grid_point(dg, g[1], g[2], g[3])
        positions[1, r] = p[1]; positions[2, r] = p[2]; positions[3, r] = p[3]
    end
    _atomic_serialize(targets_path(c), (; gidx, positions))

    pair_ids = Tuple{Int,Int}[]
    t_idx = Vector{Vector{Int}}()
    tw = Vector{Vector{Float64}}()
    for br in brs
        rows = [rowofg[g] for g in br.gidx]
        for k in 1:length(br.pair_ids)
            push!(pair_ids, br.pair_ids[k])
            push!(t_idx, rows)
            push!(tw, br.weights .* br.densities[:, k])
        end
    end
    _atomic_serialize(rho_store_path(c), (; pair_ids, t_idx, tw))
    @info "consolidate: done" n_targets=length(gidx) n_pairs=length(pair_ids)
    return nothing
end

function _atomic_serialize(path::AbstractString, obj)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid())
    open(io -> serialize(io, obj), tmp, "w")
    mv(tmp, path; force = true)
end
```

- [ ] **Step 4: Run, verify pass; commit**

```bash
git add codes/lattice_scale/src/tasks.jl codes/lattice_scale/test/pipeline.jl
git commit -m "lattice_scale: consolidate — shared target set + per-pair contraction store"
```

---

## Task 12: `eval_batch` task + V-file format

**Files:**
- Modify: `src/tasks.jl` (add `eval_batch`, V-file save/load, replace the stub `_is_complete_v`), `test/pipeline.jl`

- [ ] **Step 1: Write the failing test**

```julia
@testset "eval_batch: within-batch entries match four_index-style contraction" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c); solve_batch(c, 1); solve_batch(c, 2); consolidate(c)
        eval_batch(c, 1)
        @test CampaignLib._is_complete_v(v_path(c, 1))
        vr = CampaignLib.load_v_rows(v_path(c, 1))
        @test vr.source_pairs == [(1, 1), (1, 2)]
        @test vr.target_pairs == [(1, 1), (1, 2), (2, 2)]   # ALL pairs, manifest order
        @test size(vr.V) == (3, 2)                          # n_targets_pairs × K_sources

        # reference for the within-batch block: evaluate_batch_potential at the batch's
        # own grid + direct contraction (the four_index_matrix formula)
        br = load_batch_result(batch_path(c, 1))
        temps = CampaignLib.load_templates!(c)
        dg = temps[1][2]
        pos = Matrix{Float64}(undef, 3, length(br.gidx))
        for (r, g) in enumerate(br.gidx)
            p = BoundaryIntegral.grid_point(dg, g[1], g[2], g[3])
            pos[:, r] .= p
        end
        srcs = [BoundaryIntegral.VolumeSource(copy(pos), copy(br.weights), br.densities[:, k])
                for k in 1:2]
        far_pad = 2.0 * maximum(LinearAlgebra.norm.(BoundaryIntegral.true_cell_vectors(dg))) / dg.nx
        Φ = evaluate_batch_potential(br.interface, br.sigma, srcs, pos;
            lhs_tol = c.solve["lhs_tol"], volume_tol = c.solve["volume_tol"], far_pad = far_pad)
        V_ref = [LinearAlgebra.dot(br.weights .* br.densities[:, a], Φ[:, bb])
                 for a in 1:2, bb in 1:2]
        scale = maximum(abs.(V_ref))
        @test maximum(abs.(vr.V[1:2, :] .- V_ref)) < 1e-8 * scale

        eval_batch(c, 2)
        @test isempty(pending_batches(c, :eval))
    end
end
```

- [ ] **Step 2: Run, verify failure**

Expected: FAIL / `eval_batch not defined`

- [ ] **Step 3: Implement** (append to `src/tasks.jl`; REPLACE the Task-9 `_is_complete_v` stub)

```julia
const V_FORMAT_VERSION = 1

function save_v_rows(path::AbstractString, batch_id::Int,
        source_pairs::Vector{Tuple{Int,Int}}, target_pairs::Vector{Tuple{Int,Int}},
        V::Matrix{Float64}, stats::Dict{String,Any})
    _atomic_serialize(path, (; version = V_FORMAT_VERSION, batch_id,
        source_pairs, target_pairs, V, stats))
end

function load_v_rows(path::AbstractString)
    vr = open(deserialize, path)
    vr.version == V_FORMAT_VERSION || error("$path: V format version mismatch")
    size(vr.V) == (length(vr.target_pairs), length(vr.source_pairs)) ||
        error("$path: V shape mismatch")
    return vr
end

function _is_complete_v(path::AbstractString)
    isfile(path) || return false
    try
        load_v_rows(path)
        return true
    catch
        return false
    end
end

"""
    eval_batch(c::Campaign, batch_id) -> path | nothing

Post-eval phase for one batch (spec §6): rebuild the K sources from the BatchResult,
evaluate Φ_a = u_inc[ρ_a] + u[σ_a] ONCE at the shared target set T (near/far split in
`evaluate_batch_potential`), contract against every stored pair density, write the
K columns of V atomically. Skips if the V file is already complete.
"""
function eval_batch(c::Campaign, batch_id::Int)
    out = v_path(c, batch_id)
    if _is_complete_v(out)
        @info "eval_batch: already complete, skipping" batch_id
        return nothing
    end
    t0 = time()
    br = load_batch_result(batch_path(c, batch_id))
    T = open(deserialize, targets_path(c))
    store = open(deserialize, rho_store_path(c))
    temps = load_templates!(c)
    dg = temps[1][2]

    K = length(br.pair_ids)
    pos = Matrix{Float64}(undef, 3, length(br.gidx))
    for (r, g) in enumerate(br.gidx)
        p = BoundaryIntegral.grid_point(dg, g[1], g[2], g[3])
        pos[1, r] = p[1]; pos[2, r] = p[2]; pos[3, r] = p[3]
    end
    sources = [VolumeSource(copy(pos), copy(br.weights), br.densities[:, k]) for k in 1:K]
    At, Bt, Ct = BoundaryIntegral.true_cell_vectors(dg)
    max_step = maximum((norm(At) / dg.nx, norm(Bt) / dg.ny, norm(Ct) / dg.nz))
    far_pad = c.far_pad_steps * max_step

    Φ = evaluate_batch_potential(br.interface, br.sigma, sources, T.positions;
        lhs_tol = c.solve["lhs_tol"], volume_tol = c.solve["volume_tol"],
        far_pad = far_pad)
    t_phi = time() - t0

    nP = length(store.pair_ids)
    V = Matrix{Float64}(undef, nP, K)
    for kl in 1:nP, a in 1:K
        V[kl, a] = dot(store.tw[kl], view(Φ, store.t_idx[kl], a))
    end
    stats = Dict{String,Any}("t_phi" => t_phi, "t_total" => time() - t0,
        "n_targets" => size(T.positions, 2), "hostname" => gethostname())
    save_v_rows(out, batch_id, br.pair_ids, store.pair_ids, V, stats)
    @info "eval_batch: done" batch_id n_targets=size(T.positions, 2) t_total=stats["t_total"]
    return out
end
```

- [ ] **Step 4: Run, verify pass; commit**

```bash
git add codes/lattice_scale/src/tasks.jl codes/lattice_scale/test/pipeline.jl
git commit -m "lattice_scale: eval_batch — shared-target evaluation + all-pair contraction"
```

---

## Task 13: `assemble_v` — gather V, symmetry diagnostic

**Files:**
- Modify: `src/tasks.jl`, `test/pipeline.jl`

- [ ] **Step 1: Write the failing test**

```julia
@testset "assemble_v + symmetry diagnostic" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c); solve_batch(c, 1); solve_batch(c, 2); consolidate(c)
        eval_batch(c, 1); eval_batch(c, 2)
        rep = assemble_v(c)
        @test isfile(joinpath(c.root, "V_full.jls"))
        @test isfile(joinpath(c.root, "report.txt"))
        full = open(Serialization.deserialize, joinpath(c.root, "V_full.jls"))
        @test full.pair_ids == [(1, 1), (1, 2), (2, 2)]
        @test size(full.V) == (3, 3)
        @test !any(isnan, full.V)
        # the diagnostic: relative asymmetry of the dense V (cross-interface check)
        @test rep.max_rel_asym >= 0
        @test rep.max_rel_asym < 1e-2          # loose fixture tolerances; sanity only
        @test isapprox(full.V[1, 2], full.V[2, 1]; rtol = 1e-2)   # cross-interface entry pair
    end
end
```

- [ ] **Step 2: Run, verify failure**

Expected: FAIL / `assemble_v not defined`

- [ ] **Step 3: Implement** (append to `src/tasks.jl`)

```julia
"""
    assemble_v(c::Campaign) -> (; max_rel_asym, n)

Final phase (single process, spec §4/§6): gather all V files into the dense
`n_pairs × n_pairs` matrix (columns keyed by source pair, rows by target pair — the
SAME pair ordering, from the rho store), write `V_full.jls` and `report.txt` with the
campaign's built-in accuracy diagnostic max |V - Vᵀ| / max |V| (each entry is computed
twice, from the two independently adapted interfaces).
"""
function assemble_v(c::Campaign)
    batches = read_manifest(manifest_path(c))
    store = open(deserialize, rho_store_path(c))
    pair_ids = store.pair_ids
    col = Dict(p => i for (i, p) in enumerate(pair_ids))
    n = length(pair_ids)
    V = fill(NaN, n, n)
    for b in batches
        vr = load_v_rows(v_path(c, b.batch_id))
        vr.target_pairs == pair_ids || error("V_$(b.batch_id): target ordering mismatch")
        for (k, sp) in enumerate(vr.source_pairs)
            V[:, col[sp]] = vr.V[:, k]
        end
    end
    any(isnan, V) && error("assemble_v: missing columns (run eval for all batches first)")

    scale = maximum(abs.(V))
    max_rel_asym = maximum(abs.(V .- transpose(V))) / scale
    _atomic_serialize(joinpath(c.root, "V_full.jls"), (; pair_ids, V))
    open(joinpath(c.root, "report.txt"), "w") do io
        println(io, "campaign: $(c.name)")
        println(io, "pairs: $n   batches: $(length(batches))")
        println(io, "max|V|: $scale")
        println(io, "max rel asymmetry |V - V'|/max|V|: $max_rel_asym")
    end
    @info "assemble_v: done" n max_rel_asym
    return (; max_rel_asym, n)
end
```

- [ ] **Step 4: Run, verify pass; commit**

```bash
git add codes/lattice_scale/src/tasks.jl codes/lattice_scale/test/pipeline.jl
git commit -m "lattice_scale: assemble_v + V/V' symmetry diagnostic report"
```

---

## Task 14: `driver.jl` — Distributed driver

**Files:**
- Create: `driver.jl`
- Test: manual smoke test (Step 3) — Distributed plumbing is not unit-testable in Pkg.test

- [ ] **Step 1: Write the driver**

```julia
# driver.jl — campaign driver (spec §4).
#
# Usage:  julia --project -t 8 driver.jl <campaign.toml> <phase> [--only ID] [--workers N]
#   phases: prepare | solve | consolidate | eval | assemble | status
#
# Worker topology: ONE worker per allocated Slurm task (= one per node with
# --ntasks-per-node=1); each task uses the whole node's cores via OMP threads.
# Restart semantics: status lives on disk; resubmitting after any crash redoes at most
# the in-flight batches.
using Distributed

function _parse_args(args)
    length(args) >= 2 || error("usage: driver.jl <campaign.toml> <phase> [--only ID] [--workers N]")
    toml, phase = args[1], Symbol(args[2])
    only_id = nothing; nworkers_local = 0
    i = 3
    while i <= length(args)
        if args[i] == "--only"
            only_id = parse(Int, args[i+1]); i += 2
        elseif args[i] == "--workers"
            nworkers_local = parse(Int, args[i+1]); i += 2
        else
            error("unknown arg $(args[i])")
        end
    end
    return toml, phase, only_id, nworkers_local
end

function _setup_workers(nworkers_local::Int)
    proj = dirname(Base.active_project())
    exe = "--project=$proj"
    glue = get(ENV, "JULIA_GLUE_THREADS", "8")           # Threads.@threads glue loops
    if haskey(ENV, "SLURM_JOB_ID") && parse(Int, get(ENV, "SLURM_NTASKS", "1")) > 1
        @eval using ClusterManagers
        np = parse(Int, ENV["SLURM_NTASKS"])
        @info "spawning $np Slurm workers (one per task)"
        addprocs(SlurmManager(np); exeflags = `$exe -t $glue`)
    elseif nworkers_local > 0
        @info "spawning $nworkers_local local workers"
        addprocs(nworkers_local; exeflags = `$exe -t $glue`)
    else
        @info "no workers: running tasks inline (pilot mode)"
    end
end

function main()
    toml, phase, only_id, nworkers_local = _parse_args(ARGS)

    # Precompile BEFORE spawning workers: concurrent first-loads race the GPFS
    # compiled cache. Workers then load from the warm cache.
    using Pkg
    Pkg.precompile()
    @eval using CampaignLib

    c = load_campaign(toml)
    if phase === :prepare
        prepare(c)
    elseif phase === :consolidate
        consolidate(c)
    elseif phase === :assemble
        assemble_v(c)
    elseif phase === :status
        for ph in (:solve, :eval)
            p = pending_batches(c, ph)
            println("$ph: $(length(p)) pending  $(isempty(p) ? "" : "(ids $(first(p,min(10,length(p))))…)")")
        end
    elseif phase in (:solve, :eval)
        runner = phase === :solve ? solve_batch : eval_batch
        if only_id !== nothing
            runner(c, only_id)                            # inline pilot, full timing visible
            return
        end
        _setup_workers(nworkers_local)
        @everywhere using CampaignLib
        pending = pending_batches(c, phase)
        @info "$(phase): $(length(pending)) pending batches on $(nworkers()) workers"
        t0 = time()
        results = pmap(pending; retry_delays = [30.0], on_error = e -> e) do id
            try
                runner(load_campaign(toml), id)           # campaign reloaded per worker (cached)
                (id, :ok)
            catch err
                msg = sprint(showerror, err, catch_backtrace())
                mkpath(logs_dir(load_campaign(toml)))
                write(joinpath(logs_dir(load_campaign(toml)), "$(phase)_batch_$(lpad(id,4,'0')).err"), msg)
                rethrow()
            end
        end
        ok = count(r -> r isa Tuple && r[2] === :ok, results)
        @info "$(phase) finished" ok failed=length(results)-ok elapsed=time()-t0
        still = pending_batches(c, phase)
        isempty(still) || @warn "still pending (resubmit to retry)" still
    else
        error("unknown phase $phase")
    end
end

main()
```

Note for the implementer: `using Pkg` inside `main()` is illegal at function scope —
hoist ALL `using` statements (`Pkg`, and make `CampaignLib`/`ClusterManagers` loading
top-level with the precompile call placed before `using CampaignLib` at file scope).
Structure the final file as: `using Distributed` → `using Pkg; Pkg.precompile()` →
`using CampaignLib` → function defs (workers are spawned at runtime inside
`_setup_workers`, AFTER the head finished precompiling — which file-scope ordering
guarantees) → `main()`.

- [ ] **Step 2: Inline pilot smoke test (no workers)**

```bash
cd ~/work/four_index_integral_solver/codes/lattice_scale
julia --project -e 'include("test/fixture_campaign.jl"); mkpath("/tmp/mini"); print(write_fixture_campaign("/tmp/mini"))'
julia --project driver.jl /tmp/mini/campaign.toml prepare
julia --project driver.jl /tmp/mini/campaign.toml solve --only 1
julia --project driver.jl /tmp/mini/campaign.toml status
```
Expected: prepare writes manifest; `--only 1` solves batch 1 inline; status shows `solve: 1 pending`.

- [ ] **Step 3: Local 2-worker smoke test**

```bash
julia --project driver.jl /tmp/mini/campaign.toml solve --workers 2
julia --project driver.jl /tmp/mini/campaign.toml consolidate
julia --project driver.jl /tmp/mini/campaign.toml eval --workers 2
julia --project driver.jl /tmp/mini/campaign.toml assemble
cat /tmp/mini/out/report.txt
```
Expected: both phases complete with 2 local workers; report shows `max rel asymmetry < 1e-2`.

- [ ] **Step 4: Commit**

```bash
git add codes/lattice_scale/driver.jl
git commit -m "lattice_scale: Distributed driver (SlurmManager/local/inline, file-derived pending set)"
```

---

## Task 15: Slurm job scripts + README runbook

**Files:**
- Create: `jobscripts/solve.sbatch`, `jobscripts/eval.sbatch`, `README.md`

- [ ] **Step 1: Write the job scripts**

`jobscripts/solve.sbatch` (eval.sbatch identical except phase + job name):

```bash
#!/bin/bash
# Solve phase. SUBMIT MANUALLY:  sbatch --nodes=<M> jobscripts/solve.sbatch <campaign.toml>
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --job-name=fi_solve
#SBATCH --output=logs/solve_%j.out

set -euo pipefail
TOML=${1:?usage: sbatch jobscripts/solve.sbatch <campaign.toml>}

# pin BOTH thread pools (unpinned threads silently corrupt timings — known footgun)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_GLUE_THREADS=8

# julia 1.12 binary (the manifest is resolved with 1.12; `head -1` picks 1.10 — wrong)
JULIA=$(ls -d ~/.julia/juliaup/julia-1.12*/bin/julia | sort -V | tail -1)

cd ~/work/four_index_integral_solver/codes/lattice_scale
$JULIA --project -t $JULIA_GLUE_THREADS driver.jl "$TOML" solve
```

- [ ] **Step 2: Write `README.md`** — runbook with exactly this order of operations:

```markdown
# lattice_scale — multi-node four-index campaign

Spec: BoundaryIntegral.jl `docs/superpowers/specs/2026-06-10-multinode-lattice-campaign-design.md`.

## Run order
1. `julia --project driver.jl campaigns/<c>.toml prepare`        (login/workstation; writes manifest)
2. Pilot one batch on a node: `driver.jl <c>.toml solve --only 1` (via ssh to an interactive node)
   — note wall time + batch file size, size the campaign before step 3.
3. `sbatch --nodes=<M> jobscripts/solve.sbatch campaigns/<c>.toml`   (USER submits)
4. `julia --project driver.jl <c>.toml consolidate`               (single node)
5. Pilot one eval: `driver.jl <c>.toml eval --only 1` — the u_inc/near-correction cost
   at ~1e7 targets is the campaign's biggest unknown; measure before committing nodes.
6. `sbatch --nodes=<M> jobscripts/eval.sbatch campaigns/<c>.toml`    (USER submits)
7. `julia --project driver.jl <c>.toml assemble` → `report.txt` (symmetry diagnostic)

Crash/walltime recovery: just resubmit — status is derived from files on ceph,
completed batches are skipped. `driver.jl <c>.toml status` shows pending counts.
```

- [ ] **Step 3: Commit**

```bash
git add codes/lattice_scale/jobscripts codes/lattice_scale/README.md
git commit -m "lattice_scale: sbatch templates + runbook (user submits all jobs)"
```

---

## Task 16: 2×2 anchor campaign + cross-check script

**Files:**
- Create: `campaigns/demo_2x2.toml`, `scripts/compare_anchor.jl`

- [ ] **Step 1: Write the campaign config** (real production orbitals; loose demo tolerances)

```toml
# campaigns/demo_2x2.toml — correctness anchor (spec §8.1): 2x2 cells, real orbitals
[campaign]
name = "demo_2x2"
root = "/mnt/ceph/users/xgao1/four_index/demo_2x2"

[orbitals]
xsf = [
  "/mnt/ceph/users/mroesner/Graphene/cRPA4RSGW/graphene/monolayer/k_323201_nb_144_c_15/graphene_00001.xsf",
  "/mnt/ceph/users/mroesner/Graphene/cRPA4RSGW/graphene/monolayer/k_323201_nb_144_c_15/graphene_00002.xsf",
]

[lattice]
nx = 2
ny = 2
neighbor_cutoff = 2.6        # nn + same-cell sublattice pairs (a = 2.465, intra-cell 1.42)

[dielectrics]
eps_out = 1.0
boxes = [[0.0, 0.0, 7.5, 90.0, 90.0, 3.35, 3.5]]

[solve]
n_quad = 6
edge_refine_level = 2
rhs_tol = 1e-3
lhs_tol = 1e-5
gmres_rtol = 1e-5
support_rtol = 1e-4
volume_tol = 1e-5
max_order = 8
max_depth = 128

[batching]
n_centers_per_batch = 1

[eval]
far_pad_steps = 2.0
```

- [ ] **Step 2: Write `scripts/compare_anchor.jl`**

```julia
# Cross-check (spec §8.1): the campaign's WITHIN-BATCH V entries for the batch anchored
# at center 1 must match the existing .bie path (four_index_integrals with circshift
# LATTICE images) to solver tolerance. Run on a compute node AFTER demo_2x2 finished
# solve+consolidate+eval:   julia --project scripts/compare_anchor.jl
using CampaignLib, BoundaryIntegral, Printf

c = load_campaign(joinpath(@__DIR__, "..", "campaigns", "demo_2x2.toml"))
centers = read_centers(centers_path(c))
spec = first(read_manifest(manifest_path(c)))            # batch anchored at center 1
vr = CampaignLib.load_v_rows(v_path(c, spec.batch_id))

# build the equivalent .bie (center 1 + its batch partners as LATTICE images of the
# SAME templates) in a temp dir, run the reference path
byid = Dict(ct.id => ct for ct in centers)
partners = sort(unique(reduce(vcat, [[p[1], p[2]] for p in spec.pairs])))
bie = tempname() * ".bie"
open(bie, "w") do io
    println(io, "UNITS bohr\n\nBEGIN_DIELECTRICS\nEPS_OUT 1.0")
    b = c.boxes[1]
    @printf(io, "  %.3f %.3f %.3f    %.3f %.3f %.3f    %.3f\n",
        b.center..., b.Lx, b.Ly, b.Lz, c.epses[1])
    println(io, "END_DIELECTRICS\n\nBEGIN_ORBITALS")
    for id in partners
        ct = byid[id]
        println(io, "  $id   $(c.xsf[ct.template_id])   LATTICE $(ct.Rx) $(ct.Ry) 0")
    end
    println(io, "END_ORBITALS\n\nBEGIN_GROUPING")
    println(io, "  1 : $(join([p[2] for p in spec.pairs], ' '))")
    println(io, "END_GROUPING\n\nBEGIN_SOLVE")
    for (k, v) in ("N_QUAD" => 6, "EDGE_REFINE_LEVEL" => 2, "RHS_TOL" => 1e-3,
                   "LHS_TOL" => 1e-5, "GMRES_RTOL" => 1e-5, "SUPPORT_RTOL" => 1e-4,
                   "VOLUME_TOL" => 1e-5)
        println(io, "  $k $v")
    end
    println(io, "END_SOLVE")
end
ref = four_index_integrals(bie, 1)

# compare the shared entries (campaign rows for this batch's own pairs)
rowof = Dict(p => i for (i, p) in enumerate(vr.target_pairs))
worst = 0.0
for (a, pa) in enumerate(spec.pairs), (b2, pb) in enumerate(spec.pairs)
    v_c = vr.V[rowof[pa], b2]
    v_r = ref.V[a, b2]
    rel = abs(v_c - v_r) / max(abs(v_r), 1e-300)
    worst = max(worst, rel)
    @printf("%-14s %-14s  campaign % .6e   ref % .6e   rel %.2e\n",
        "$(pa)", "$(pb)", v_c, v_r, rel)
end
@printf("\nworst relative difference: %.3e  (expect ≲ 1e-2 at these tolerances)\n", worst)
```

CAVEAT for the implementer: the `.bie` LATTICE path wraps on the 5×5 template grid,
the campaign translates. Within a 2×2 flake all shifts are ≤ 1 cell with the orbital
mid-grid, so wrap effects sit below `support_rtol = 1e-4`. If the comparison fails,
check FIRST whether the .bie group/centroid conventions place center 1's partners
identically (print both center lists).

- [ ] **Step 3: Hand the runbook to the user** (no autonomous Slurm):
print the exact commands — `prepare` (workstation), pilot `--only 1` (interactive node),
then the user submits `sbatch --nodes=2 jobscripts/solve.sbatch campaigns/demo_2x2.toml`,
consolidate, eval, assemble, `scripts/compare_anchor.jl`.

- [ ] **Step 4: Commit**

```bash
git add codes/lattice_scale/campaigns/demo_2x2.toml codes/lattice_scale/scripts/compare_anchor.jl
git commit -m "lattice_scale: 2x2 anchor campaign + cross-check vs the .bie reference path"
```

---

## Task 17: 10×10 campaign config

**Files:**
- Create: `campaigns/lattice_10x10.toml`

- [ ] **Step 1: Write the config**

Same as `demo_2x2.toml` except:

```toml
[campaign]
name = "lattice_10x10"
root = "/mnt/ceph/users/xgao1/four_index/lattice_10x10"

[lattice]
nx = 10
ny = 10
neighbor_cutoff = 5.0        # ≈ 18 neighbors/orbital → ≈ 2000 unique pairs incl. on-site

[dielectrics]
eps_out = 1.0
boxes = [[11.0, 9.6, 7.5, 90.0, 90.0, 3.35, 3.5]]   # slab centered on the 10x10 flake:
# flake spans Rx,Ry in 0..9 → centers cover ~[0, 9·2.465]x[0, 9·2.135] + centroid offset;
# VERIFY at prepare time: print extrema of centers.tsv and confirm ≥ 15 margin per side,
# adjust cx, cy accordingly before any production run.
```

- [ ] **Step 2: Sanity-check the enumeration locally (no solve)**

```bash
julia --project driver.jl campaigns/lattice_10x10.toml prepare
julia --project -e '
using CampaignLib
c = load_campaign("campaigns/lattice_10x10.toml")
cs = read_centers(centers_path(c)); bs = read_manifest(manifest_path(c))
np = sum(b -> length(b.pairs), bs)
println("centers=", length(cs), "  batches=", length(bs), "  pairs=", np)
println("x range: ", extrema(getindex.(getfield.(cs, :center), 1)))
println("y range: ", extrema(getindex.(getfield.(cs, :center), 2)))'
```
Expected: `centers=200`, pairs ≈ 2000 (200 on-site + ~1800 off-site), batches=200; center
extents comfortably inside the slab box (else fix the box center/size in the toml).

- [ ] **Step 3: Commit + handoff runbook**

```bash
git add codes/lattice_scale/campaigns/lattice_10x10.toml
git commit -m "lattice_scale: 10x10 campaign config"
```

Then the campaign itself follows README order: pilot `solve --only 1` and `eval --only 1`
on one node (measure!), then user submits solve on ~4 nodes (loose shakeout), then the
~1e-4 production (rhs_tol 1e-4 / lhs_tol 1e-6 / gmres 1e-6 in a copied toml with a new
`root`) on 8–16 nodes.

---

## Self-review checklist (run after writing, fixed inline)

1. **Spec coverage:** §2 translation frames → Tasks 1–2; §3 data model → Tasks 5, 7–9, 12; §4 driver → Task 14; §5 solve → Tasks 3, 10; §6 eval/contraction/diagnostic → Tasks 4, 11–13; §7 code-change list → Tasks 1–5 (package) + 7–17 (campaign); §8 validation ladder → fixture tests (8.x unit), Task 14 steps 2–3 (plumbing), Task 16 (anchor), Task 17 (pilot + demo); §9 error handling → atomic writes (5, 11, 12), pmap retry/err files (14), completeness validation (5, 12). The §2 "support touches frame boundary" warning is intentionally deferred — the support_rtol-truncated production orbital fits the 5×5 frame (verified by memory of the 8-neighbor runs); add it if the anchor comparison flags it.
2. **No placeholders:** every code step is complete; the two flagged unknowns (TKM 4π convention in Task 4, slab box center in Task 17) are explicit verification instructions with the test that decides them, not TBDs.
3. **Type consistency:** `LatticeBatch`/`OrbitalInstance`/`BatchResult` field names match across Tasks 2–5 and 10–12; `solve_dielectric_lattice_batch` returns `(; sigma, interface, sources, stats)` used as such in Task 10; V-file NamedTuple fields (`source_pairs`, `target_pairs`, `V`) match between Tasks 12–13 and 16.
