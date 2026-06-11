# TOML Campaign Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `.bie` with a unified `.toml` input, fold `CampaignLib` into `BoundaryIntegral.jl` (core + a Distributed/SlurmClusterManager package extension), make orbital placement an explicit snap-to-grid list, and emit the four-index result as a text table.

**Architecture:** A dependency-free `src/campaign/` core (TOML parsing → `CampaignInput`, snap-to-grid geometry, manifest, in-memory + file-based phase functions, text output) plus `ext/BoundaryIntegralDistributedExt.jl` (parallel driver via `SlurmClusterManager`). The work repo keeps only a thin deployment layer. `.bie`/`SystemInput` is deleted; `four_index_matrix` is decoupled from `SystemInput` and kept as an independent test reference.

**Tech Stack:** Julia 1.12, TOML stdlib, the existing `lattice_batch.jl`/`batch_io.jl` machinery (`OrbitalInstance`, `assemble_lattice_batch`, `solve_dielectric_lattice_batch`, `evaluate_batch_potential`, `BatchResult`), Distributed + SlurmClusterManager (weakdeps).

**Spec:** `docs/superpowers/specs/2026-06-10-toml-campaign-unification-design.md`

**Conventions:**
- Package repo: `/mnt/home/xgao1/codes/BoundaryIntegral.jl`, branch `multi_rhs`. Work repo: `/mnt/home/xgao1/work/four_index_integral_solver` (campaign deployment under `codes/lattice_scale/`).
- Run Julia as `julia` (juliaup 1.12 at `/mnt/home/xgao1/.juliaup/bin/julia`; NEVER `module load julia`). Non-interactive ssh shells need the absolute path.
- NEVER execute Slurm submit commands (`sbatch`/`srun`/`salloc`) — those are the user's. Read-only `squeue`/`sacct`/`seff` are allowed.
- Package tests run standalone: `julia --project=. test/<file>.jl`, full suite `julia --project=. -e 'using Pkg; Pkg.test()'`.
- Commits go on `multi_rhs` (package) / the work repo's `main` (deployment), as in prior tasks.

**Current state to build on (already present from Tasks 1–13):**
- `src/solver/lattice_batch.jl`: `OrbitalInstance(id, template_id, steps::NTuple{3,Int})`, `lattice_grid_steps(dg, primvec, n)`, `assemble_lattice_batch(grids, instances::Dict{Int,OrbitalInstance}, pairs; support_rtol)`, `solve_dielectric_lattice_batch(boxes, epses, eps_out, b; n_quad, rhs_atol, l_ec, fmm_tol, up_tol, max_order, gmres_rtol, max_depth)`, `evaluate_batch_potential(interface, Σ, sources, targets; lhs_tol, volume_tol, far_pad)`.
- `src/solver/batch_io.jl`: `BatchResult(version, batch_id, pair_ids, gidx, weights, densities, interface, sigma, stats)`, `save_batch_result`, `load_batch_result`, `is_complete_batch`, `BATCH_FORMAT_VERSION`.
- `CampaignLib` (work repo, to be folded in): `src/config.jl`, `src/manifest.jl`, `src/tasks.jl`.

---

## PHASE A — `.toml` format swap + delete `.bie` (riskiest first; ends with `Pkg.test()` green)

### Task 1: `CampaignInput` + `.toml` loader + path helpers

**Files:**
- Create: `src/campaign/toml_input.jl`
- Modify: `src/BoundaryIntegral.jl` (add `using TOML`; `include("campaign/toml_input.jl")` after `solver/batch_io.jl`; exports)
- Create: `test/fixtures/system_small.toml`
- Create: `test/campaign/toml_input.jl`
- Modify: `test/runtests.jl` (add `include("campaign/toml_input.jl")`)
- Modify: `Project.toml` (add `TOML` stdlib to `[deps]` + `[compat]`)

- [ ] **Step 1: Add the TOML dependency**

```bash
cd /mnt/home/xgao1/codes/BoundaryIntegral.jl
julia --project=. -e 'using Pkg; Pkg.add("TOML")'
```
Then in `Project.toml` `[compat]` add `TOML = "1.11.0"` (next to `Serialization`). Verify `julia --project=. -e 'using Pkg; Pkg.status()'` resolves.

- [ ] **Step 2: Create the test fixture `test/fixtures/system_small.toml`**

(Mirrors the deleted `system_small.bie`: 2 orbitals on a 4³ eps=11.7 box, but in the new explicit format. `orb_a.xsf` centroid is (1,1,1); place orbital 1 there, orbital 2 at (5,1,1) — i.e. orb_b shifted; cutoff 3.0 keeps them as separate on-site groups.)

```toml
name = "system_small"
root = "/tmp/system_small_unused"
templates = ["orb_a.xsf", "orb_b.xsf"]

[[orbital]]
type = 1
x = 1.0
y = 1.0
z = 1.0

[[orbital]]
type = 2
x = 5.0
y = 1.0
z = 1.0

[pairing]
neighbor_cutoff = 3.0

[dielectrics]
eps_out = 1.0
boxes = [[0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 11.7]]

[solve]
n_quad = 4
l_ec = 2.0
rhs_tol = 1e-2
lhs_tol = 1e-6
gmres_rtol = 1e-8
support_rtol = 1e-6
volume_tol = 1e-6
max_order = 8
max_depth = 128

[batching]
n_centers_per_batch = 1

[eval]
far_pad_steps = 2.0
```

- [ ] **Step 3: Write the failing test `test/campaign/toml_input.jl`**

```julia
using BoundaryIntegral
using Test

@testset "toml_input" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    c = load_campaign(joinpath(fixdir, "system_small.toml"))

    @test c.name == "system_small"
    @test length(c.templates) == 2
    @test endswith(c.templates[1], "orb_a.xsf")
    @test length(c.orbitals) == 2
    @test c.orbitals[1].type == 1 && c.orbitals[1].pos == (1.0, 1.0, 1.0)
    @test c.orbitals[2].type == 2 && c.orbitals[2].pos == (5.0, 1.0, 1.0)
    @test c.neighbor_cutoff == 3.0
    @test c.pair_overrides === nothing
    @test length(c.boxes) == 1 && c.boxes[1].Lx == 4.0 && c.epses == [11.7]
    @test c.eps_out == 1.0
    @test c.solve["n_quad"] == 4 && c.solve["l_ec"] == 2.0
    @test c.n_centers_per_batch == 1
    @test c.far_pad_steps == 2.0

    # template paths are resolved relative to the toml's directory; toml_path recorded
    @test isabspath(c.templates[1])
    @test isabspath(c.toml_path) && endswith(c.toml_path, "system_small.toml")

    # path helpers
    @test endswith(batch_path(c, 7), "batches/batch_0007.jls")
    @test endswith(v_path(c, 12), "V/V_0012.jls")
    @test endswith(manifest_path(c), "manifest.tsv")
    @test BoundaryIntegral.campaign_l_ec(c) ≈ 2.0     # l_ec given directly

    # explicit pair overrides parse when present
    mktempdir() do dir
        p = joinpath(dir, "ov.toml")
        cp(joinpath(fixdir, "system_small.toml"), p)
        open(p, "a") do io; println(io, "\n[pairing]\npairs = [[1,1],[1,2]]"); end
        # NOTE: a second [pairing] table is illegal TOML; instead test via a dedicated fixture below
    end
end
```

(Remove the `mktempdir` block — it's illustrative; replace with a dedicated override fixture if desired. Keep the rest.)

- [ ] **Step 4: Run, verify failure**

Run: `julia --project=. test/campaign/toml_input.jl`
Expected: FAIL / `load_campaign not defined`

- [ ] **Step 5: Implement `src/campaign/toml_input.jl`**

```julia
# Unified .toml input for the four-index campaign (spec
# docs/.../2026-06-10-toml-campaign-unification-design.md). Replaces the .bie format.

using TOML

"""
    OrbitalSpec(type, pos)

One orbital from the .toml `[[orbital]]` list: `type` indexes `CampaignInput.templates`
(1-based), `pos` is its Cartesian center in the templates' frame.
"""
struct OrbitalSpec
    type::Int
    pos::NTuple{3,Float64}
end

"""
    CampaignInput

Parsed `.toml`: the full system (orbitals, dielectrics, solve params) plus campaign
batching/eval settings. Successor to both `Campaign` and the deleted `SystemInput`.
Template grids are NOT loaded here (see `load_templates!`). Orbital id = 1-based index
in `orbitals`.
"""
struct CampaignInput
    name::String
    root::String
    templates::Vector{String}                 # type index → xsf path (absolute)
    orbitals::Vector{OrbitalSpec}
    neighbor_cutoff::Float64
    pair_overrides::Union{Nothing,Vector{Tuple{Int,Int}}}
    eps_out::Float64
    boxes::Vector{BoxGeom}
    epses::Vector{Float64}
    solve::Dict{String,Float64}
    n_centers_per_batch::Int
    far_pad_steps::Float64
    toml_path::String                         # absolute path of the source .toml (workers reload from it)
end

function load_campaign(toml_path::AbstractString)
    toml_path = abspath(toml_path)
    d = TOML.parsefile(toml_path)
    base = dirname(toml_path)
    haskey(d, "batching") || error("$toml_path: missing [batching] section")
    haskey(d, "orbital") || error("$toml_path: needs at least one [[orbital]] entry")

    templates = String[isabspath(t) ? t : joinpath(base, t) for t in d["templates"]]
    orbitals = OrbitalSpec[]
    for o in d["orbital"]
        push!(orbitals, OrbitalSpec(Int(o["type"]),
            (Float64(o["x"]), Float64(o["y"]), Float64(o["z"]))))
    end

    pr = get(d, "pairing", Dict())
    cutoff = Float64(get(pr, "neighbor_cutoff", Inf))
    overrides = haskey(pr, "pairs") ?
        Tuple{Int,Int}[(Int(p[1]), Int(p[2])) for p in pr["pairs"]] : nothing

    di = d["dielectrics"]
    boxes = BoxGeom[]; epses = Float64[]
    for row in di["boxes"]
        length(row) == 7 || error("dielectrics.boxes rows are [cx cy cz Lx Ly Lz eps]")
        push!(boxes, (center = (Float64(row[1]), Float64(row[2]), Float64(row[3])),
                      Lx = Float64(row[4]), Ly = Float64(row[5]), Lz = Float64(row[6])))
        push!(epses, Float64(row[7]))
    end

    solve = Dict{String,Float64}(k => Float64(v) for (k, v) in d["solve"])
    return CampaignInput(d["name"], d["root"], templates, orbitals, cutoff, overrides,
        Float64(get(di, "eps_out", 1.0)), boxes, epses, solve,
        Int(d["batching"]["n_centers_per_batch"]),
        Float64(get(get(d, "eval", Dict()), "far_pad_steps", 2.0)), toml_path)
end

manifest_path(c::CampaignInput)  = joinpath(c.root, "manifest.tsv")
centers_path(c::CampaignInput)   = joinpath(c.root, "centers.tsv")
targets_path(c::CampaignInput)   = joinpath(c.root, "targets.jls")
rho_store_path(c::CampaignInput) = joinpath(c.root, "rho_store.jls")
logs_dir(c::CampaignInput)       = joinpath(c.root, "logs")
batch_path(c::CampaignInput, id::Int) = joinpath(c.root, "batches", _b4("batch", id))
v_path(c::CampaignInput, id::Int)     = joinpath(c.root, "V", _b4("V", id))
_b4(stem, id) = string(stem, "_", lpad(id, 4, '0'), ".jls")

"Edge/corner refinement target size: `solve.l_ec` if set, else min box Lz / 2^level · 1.01."
function campaign_l_ec(c::CampaignInput)
    haskey(c.solve, "l_ec") && return c.solve["l_ec"]
    level = Int(get(c.solve, "edge_refine_level", 4))
    minimum(b.Lz for b in c.boxes) / 2.0^level * 1.01
end

# worker-local template cache: path => (structure, datagrid)
const TEMPLATE_CACHE = Dict{String,Any}()
function load_templates!(c::CampaignInput)
    return [get!(TEMPLATE_CACHE, p) do
                read_xsf(p)
            end for p in c.templates]
end
```

In `src/BoundaryIntegral.jl`: add `using TOML` near the other `using`s; `include("campaign/toml_input.jl")` after `include("solver/batch_io.jl")`; export `CampaignInput, OrbitalSpec, load_campaign, manifest_path, centers_path, targets_path, rho_store_path, logs_dir, batch_path, v_path`. (`campaign_l_ec`, `load_templates!`, `TEMPLATE_CACHE` stay unexported.) Add `include("campaign/toml_input.jl")` test to `test/runtests.jl` via `include("campaign/toml_input.jl")` (create the `test/campaign/` dir).

- [ ] **Step 6: Run, verify pass**

Run: `julia --project=. test/campaign/toml_input.jl`
Expected: PASS (after removing the illustrative `mktempdir` block from the test)

- [ ] **Step 7: Commit**

```bash
git add src/campaign/toml_input.jl src/BoundaryIntegral.jl test/fixtures/system_small.toml test/campaign/toml_input.jl test/runtests.jl Project.toml
git commit -m "feat: CampaignInput + .toml loader (campaign/toml_input.jl)"
```

---

### Task 2: snap-to-grid geometry

**Files:**
- Create: `src/campaign/geometry.jl`
- Modify: `src/BoundaryIntegral.jl` (include after `toml_input.jl`; export `snap_orbital`)
- Test: `test/campaign/geometry.jl` (create); add to `test/runtests.jl`

- [ ] **Step 1: Write the failing test**

```julia
using BoundaryIntegral
using Test

@testset "geometry snap" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    # orb_lat: 4x4x4, true cell 2.0, step 0.5; primvec = I => 2 steps per unit length.
    st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_lat.xsf"))
    c0 = BoundaryIntegral.density_centroid(dg)   # spike at origin -> (0,0,0)

    # exact lattice point: centroid + (1,0,0) -> 2 steps along x
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1] + 1.0, c0[2], c0[3])) == (2, 0, 0)
    @test BoundaryIntegral.snap_orbital(dg, c0, c0) == (0, 0, 0)
    # off-lattice by < half a step (0.25 < 0.5/2 in step units) snaps to nearest
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1] + 0.24, c0[2], c0[3])) == (0, 0, 0)
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1] + 0.26, c0[2], c0[3])) == (1, 0, 0)
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1], c0[2], c0[3] - 1.0)) == (0, 0, -2)
end

@testset "geometry snap: hexagonal skew (orb_smooth uses I; use a skew check via grid basis)" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    st, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_smooth.xsf"))
    c0 = BoundaryIntegral.density_centroid(dg)
    # one cell along grid axis 1: true cell 3.0 over 6 pts (step 0.5), 2 steps/unit, 6 steps/cell? 
    # orb_smooth: span 2.5, true cell 3.0, nx=6 -> step 0.5 -> a displacement of +0.5 = 1 step.
    @test BoundaryIntegral.snap_orbital(dg, c0, (c0[1] + 0.5, c0[2], c0[3])) == (1, 0, 0)
end
```

- [ ] **Step 2: Run, verify failure**

Run: `julia --project=. test/campaign/geometry.jl`
Expected: FAIL / `snap_orbital not defined`

- [ ] **Step 3: Implement `src/campaign/geometry.jl`**

```julia
# Snap an arbitrary Cartesian orbital position to the nearest integer grid offset on the
# template's grid, preserving the exact commensurate-grid machinery (spec §4).

using LinearAlgebra

"""
    snap_orbital(datagrid, centroid, pos) -> NTuple{3,Int}

Integer grid-step offset that places an orbital (template `datagrid`, density centroid
`centroid`) closest to Cartesian `pos`. Solves `pos - centroid = G * s` for real `s` in
the grid-step basis `G = [At/nx Bt/ny Ct/nz]` (handles non-orthogonal a1/a2), then rounds.
Warns if the snap moves the center by more than half the largest grid step; errors if the
z-offset is not (near-)integer (a planar campaign — a fractional z is almost certainly a
frame/unit mistake).
"""
function snap_orbital(datagrid, centroid::NTuple{3,Float64}, pos::NTuple{3,Float64})
    At, Bt, Ct = true_cell_vectors(datagrid)
    G = hcat(collect(At) ./ datagrid.nx, collect(Bt) ./ datagrid.ny, collect(Ct) ./ datagrid.nz)
    rhs = collect(pos) .- collect(centroid)
    s = G \ rhs                                  # real fractional steps
    steps = round.(Int, s)
    realized = collect(centroid) .+ G * steps
    max_step = maximum(norm.((G[:, 1], G[:, 2], G[:, 3])))
    if norm(rhs .- G * steps) > 0.5 * max_step
        @warn "snap_orbital: orbital snapped >½ grid step" requested=pos realized=Tuple(realized)
    end
    abs(s[3] - steps[3]) <= 1e-3 ||
        error("snap_orbital: z=$(pos[3]) is not representable on the template grid (Δ=$(s[3]-steps[3]) steps)")
    return (steps[1], steps[2], steps[3])
end
```

In `src/BoundaryIntegral.jl`: `include("campaign/geometry.jl")` after `toml_input.jl`; export `snap_orbital`. Add `include("campaign/geometry.jl")` to `test/runtests.jl`.

- [ ] **Step 4: Run, verify pass**

Run: `julia --project=. test/campaign/geometry.jl`
Expected: PASS. (If the orb_smooth step count assumption is off, adjust the displacement in the test to one true grid step = `norm(At)/nx`; the implementation is the authority — do not change the rounding logic.)

- [ ] **Step 5: Commit**

```bash
git add src/campaign/geometry.jl src/BoundaryIntegral.jl test/campaign/geometry.jl test/runtests.jl
git commit -m "feat: snap_orbital — Cartesian orbital position to nearest integer grid offset"
```

---

### Task 3: manifest (explicit-orbital `CenterInfo`, no Rx/Ry)

**Files:**
- Create: `src/campaign/manifest.jl` (adapted from CampaignLib's `manifest.jl`)
- Modify: `src/BoundaryIntegral.jl` (include; exports)
- Test: `test/campaign/manifest.jl` (create); add to `test/runtests.jl`

- [ ] **Step 1: Write the failing test**

```julia
using BoundaryIntegral
using Test

@testset "manifest" begin
    centers = BoundaryIntegral.CenterInfo[
        BoundaryIntegral.CenterInfo(1, 1, (0,0,0),   (0.0, 0.0, 0.0)),
        BoundaryIntegral.CenterInfo(2, 2, (0,0,0),   (0.5, 0.5, 0.0)),
        BoundaryIntegral.CenterInfo(3, 1, (2,0,0),   (1.0, 0.0, 0.0)),
    ]
    pairs = enumerate_pairs(centers, 0.8)        # on-site + the d=0.707 (1,2) pair; not (1,3)@1.0
    @test all(p -> p[1] <= p[2], pairs)
    @test (1, 1) in pairs && (2, 2) in pairs && (3, 3) in pairs && (1, 2) in pairs
    @test !((1, 3) in pairs)

    batches = build_batches(pairs, 1)
    @test sum(b -> length(b.pairs), batches) == length(pairs)
    @test allunique(getfield.(batches, :batch_id))

    mktempdir() do dir
        cp = joinpath(dir, "centers.tsv"); mp = joinpath(dir, "manifest.tsv")
        write_centers(cp, centers); write_manifest(mp, batches)
        @test read_centers(cp) == centers
        @test read_manifest(mp) == batches
    end
end
```

- [ ] **Step 2: Run, verify failure**

Run: `julia --project=. test/campaign/manifest.jl`
Expected: FAIL / `CenterInfo not defined`

- [ ] **Step 3: Implement `src/campaign/manifest.jl`**

(Same as CampaignLib's `manifest.jl` EXCEPT: `CenterInfo` drops `Rx`/`Ry`; `enumerate_centers` is replaced by `enumerate_centers(c::CampaignInput)` which snaps the explicit orbital list; the centers TSV drops the `Rx`/`Ry` columns. `enumerate_pairs`, `build_batches`, `write_manifest`/`read_manifest` are unchanged from CampaignLib.)

```julia
using Printf

"""
    CenterInfo(id, template_id, steps, center)

One orbital site: 1-based `id`, `template_id` (index into `CampaignInput.templates`), the
integer global-grid offset `steps`, and the realized Cartesian `center`.
"""
struct CenterInfo
    id::Int
    template_id::Int
    steps::NTuple{3,Int}
    center::NTuple{3,Float64}
end

struct BatchSpec
    batch_id::Int
    anchors::Vector{Int}
    pairs::Vector{Tuple{Int,Int}}
end

Base.:(==)(a::CenterInfo, b::CenterInfo) =
    a.id == b.id && a.template_id == b.template_id && a.steps == b.steps && a.center == b.center
Base.:(==)(a::BatchSpec, b::BatchSpec) =
    a.batch_id == b.batch_id && a.anchors == b.anchors && a.pairs == b.pairs

"""
    enumerate_centers(c::CampaignInput) -> Vector{CenterInfo}

Snap each explicit `[[orbital]]` to its integer grid offset (against its template's grid)
and record the realized Cartesian center. Orbital id = 1-based index in `c.orbitals`.
"""
function enumerate_centers(c::CampaignInput)
    temps = load_templates!(c)
    out = CenterInfo[]
    for (id, o) in enumerate(c.orbitals)
        dg = temps[o.type][2]
        c0 = ntuple(d -> Float64(density_centroid(dg)[d]), 3)
        steps = snap_orbital(dg, c0, o.pos)
        At, Bt, Ct = true_cell_vectors(dg)
        G = hcat(collect(At) ./ dg.nx, collect(Bt) ./ dg.ny, collect(Ct) ./ dg.nz)
        realized = collect(c0) .+ G * collect(steps)
        push!(out, CenterInfo(id, o.type, steps, (realized[1], realized[2], realized[3])))
    end
    return out
end

"Unique pairs (i ≤ j) with center distance ≤ cutoff. On-site pairs (i,i) included."
function enumerate_pairs(centers::Vector{CenterInfo}, cutoff::Real)
    byid = sort(centers; by = c -> c.id)
    pairs = Tuple{Int,Int}[]
    for (m, ci) in enumerate(byid), cj in byid[m:end]
        sqrt(sum(abs2, ci.center .- cj.center)) <= cutoff && push!(pairs, (ci.id, cj.id))
    end
    return pairs
end

"""
    build_batches(pairs, n_centers_per_batch) -> Vector{BatchSpec}

Each pair belongs to its anchor (= min id); consecutive anchors merged n at a time.
"""
function build_batches(pairs::Vector{Tuple{Int,Int}}, n_centers_per_batch::Int)
    by_anchor = Dict{Int,Vector{Tuple{Int,Int}}}()
    for p in pairs
        push!(get!(by_anchor, min(p[1], p[2]), Tuple{Int,Int}[]), p)
    end
    anchors = sort(collect(keys(by_anchor)))
    out = BatchSpec[]; bid = 0
    for grp in Iterators.partition(anchors, n_centers_per_batch)
        bid += 1
        push!(out, BatchSpec(bid, collect(grp), reduce(vcat, (sort(by_anchor[a]) for a in grp))))
    end
    return out
end

function write_centers(path::AbstractString, centers::Vector{CenterInfo})
    d = dirname(path); isempty(d) || mkpath(d)
    open(path, "w") do io
        println(io, "id\ttemplate\tsx\tsy\tsz\tcx\tcy\tcz")
        for c in sort(centers; by = c -> c.id)
            int = join([c.id, c.template_id, c.steps...], '\t')
            flt = join(repr.(c.center), '\t')
            println(io, int, '\t', flt)
        end
    end
end

function read_centers(path::AbstractString)
    out = CenterInfo[]
    for (n, line) in enumerate(eachline(path))
        n == 1 && continue
        f = split(line, '\t')
        push!(out, CenterInfo(parse(Int, f[1]), parse(Int, f[2]),
            (parse(Int, f[3]), parse(Int, f[4]), parse(Int, f[5])),
            (parse(Float64, f[6]), parse(Float64, f[7]), parse(Float64, f[8]))))
    end
    return out
end

function write_manifest(path::AbstractString, batches::Vector{BatchSpec})
    d = dirname(path); isempty(d) || mkpath(d)
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
        pairs = isempty(strip(f[4])) ? Tuple{Int,Int}[] :
            [(parse(Int, split(p, ':')[1]), parse(Int, split(p, ':')[2])) for p in split(f[4], ';')]
        push!(out, BatchSpec(parse(Int, f[1]), parse.(Int, split(f[2], ',')), pairs))
    end
    return out
end
```

In `src/BoundaryIntegral.jl`: `include("campaign/manifest.jl")` after `geometry.jl`; export `CenterInfo, BatchSpec, enumerate_centers, enumerate_pairs, build_batches, write_centers, read_centers, write_manifest, read_manifest`. Add the test include.

- [ ] **Step 4: Run, verify pass**

Run: `julia --project=. test/campaign/manifest.jl` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/campaign/manifest.jl src/BoundaryIntegral.jl test/campaign/manifest.jl test/runtests.jl
git commit -m "feat: campaign manifest on explicit orbital list (CenterInfo without Rx/Ry)"
```

---

### Task 4: decouple `four_index_matrix` from `SystemInput`

**Files:**
- Modify: `src/solver/multi_rhs.jl` (change `four_index_matrix` signature)
- Test: `test/solver/lattice_batch.jl` (update the one caller)

- [ ] **Step 1: Read the current `four_index_matrix`** (`src/solver/multi_rhs.jl:475-499`). It takes `(si::SystemInput, interface, sources, Σ)` and uses only `si.solve.lhs_tol` and `si.solve.volume_tol`.

- [ ] **Step 2: Update the test caller first** in `test/solver/lattice_batch.jl` (the `"V via evaluate_batch_potential == four_index_matrix"` testset). Change:

```julia
V_ref = four_index_matrix(si_e, res_e.interface, res_e.sources, res_e.sigma)
```
to
```julia
V_ref = four_index_matrix(res_e.interface, res_e.sources, res_e.sigma;
                          lhs_tol = 1e-6, volume_tol = 1e-8)
```
(and remove the now-unused `si_e = read_system_input(...)` line in that testset, replacing any other `si_e.solve[...]` references with the literal tolerances used elsewhere in the file).

- [ ] **Step 3: Run, verify failure**

Run: `julia --project=. test/solver/lattice_batch.jl`
Expected: FAIL / `MethodError: no method matching four_index_matrix(::DielectricInterface, ...)`

- [ ] **Step 4: Change the signature** in `src/solver/multi_rhs.jl`:

```julia
"""
    four_index_matrix(interface, sources, Σ; lhs_tol, volume_tol, range_factor=5.0) -> K×K

Step 7 contraction: V[a,b] = ∫ ρ_a (u_inc[ρ_b] + u[σ_b]). Independent reference for
`evaluate_batch_potential` (different evaluation path: TKM incident + corrected-FMM pottrg
at the group grid). No longer tied to SystemInput.
"""
function four_index_matrix(interface, sources::Vector{<:VolumeSource{Float64, 3}},
        Σ::AbstractMatrix; lhs_tol::Float64, volume_tol::Float64, range_factor::Float64 = 5.0)
    K = length(sources)
    K == 0 && return zeros(Float64, 0, 0)
    targets = sources[1].positions
    pottrg = laplace3d_pottrg_fmm3d_corrected_hcubature(interface, targets, lhs_tol, lhs_tol, range_factor)
    u_inc = Vector{Vector{Float64}}(undef, K)
    for b in 1:K
        sb = screened_volume_source(interface, sources[b], SharpScreening())
        vals = TKM3D.ltkm3dc(volume_tol, sb.positions; charges = sb.weights .* sb.density,
                             targets = targets, pgt = 1, kmax = _estimate_tkm3dc_kmax(sb))
        vals.ier == 0 || error("TKM3D.ltkm3dc failed, ier=$(vals.ier)")
        u_inc[b] = real.(vals.pottarg)
    end
    tw = [sources[a].weights .* sources[a].density for a in 1:K]
    V = Matrix{Float64}(undef, K, K)
    for b in 1:K
        φb = u_inc[b] .+ (pottrg * Σ[:, b])
        for a in 1:K
            V[a, b] = dot(tw[a], φb)
        end
    end
    return V
end
```

(`four_index_matrix` stays exported.)

- [ ] **Step 5: Run, verify pass**

Run: `julia --project=. test/solver/lattice_batch.jl` → PASS (still uses the `.bie` `system_smooth_lat.bie` for the SOLVE setup — that's removed in Task 6; this task only changes `four_index_matrix`).

- [ ] **Step 6: Commit**

```bash
git add src/solver/multi_rhs.jl test/solver/lattice_batch.jl
git commit -m "refactor: decouple four_index_matrix from SystemInput (take lhs_tol/volume_tol)"
```

---

### Task 5: delete the `.bie` per-center API and `system_input.jl`

**Files:**
- Delete: `src/utils/system_input.jl`
- Modify: `src/solver/multi_rhs.jl` (remove the `SystemInput`-coupled functions)
- Modify: `src/BoundaryIntegral.jl` (remove include + exports)

- [ ] **Step 1: Delete the parser**

```bash
git rm src/utils/system_input.jl
```

- [ ] **Step 2: Remove from `src/solver/multi_rhs.jl`** these `SystemInput`-coupled definitions (keep everything else, especially `pair_density_source`, `_pair_density_array`, `_flatten_grid_array`, `_cached_xsf_grid!`, the `BatchedDielectricOperator`, `rhs_dielectric_box3d_fmm3d(interface, vss, thresh)`, `solve_dielectric_box3d_block`, `four_index_matrix`):
  - `_instance_grid(orb::OrbitalEntry, cache)` (uses `OrbitalEntry`)
  - `struct RHSGroup` + `num_pairs(::RHSGroup)` + `envelope_volume_source(::RHSGroup)` + `group_volume_sources(::RHSGroup)`
  - `assemble_rhs_group(si::SystemInput, ...)`
  - `build_group_interface(si::SystemInput, ...)`
  - `rhs_dielectric_box3d_fmm3d_batched(interface, si::SystemInput, group::RHSGroup, thresh)`
  - `solve_dielectric_box3d_group(si::SystemInput, ...)` and its `bie_path` overload
  - `four_index_integrals(si::SystemInput, ...)` and its `bie_path` overload

(These are the old per-center `.bie` path. The new path — `assemble_lattice_batch`/`solve_dielectric_lattice_batch`/`evaluate_batch_potential` in `lattice_batch.jl` — does not use any of them. `four_index_integrals(toml)` is re-added in Task 7.)

- [ ] **Step 3: Update `src/BoundaryIntegral.jl`**: remove `include("utils/system_input.jl")` and delete the export line `export read_system_input, SystemInput, OrbitalEntry, SolveParams, resolved_l_ec`. Remove `assemble_rhs_group, build_group_interface, solve_dielectric_box3d_group, four_index_integrals, RHSGroup, group_volume_sources, envelope_volume_source` from the export list **only where they refer to the deleted definitions** — note `envelope_volume_source(::LatticeBatch)` in `lattice_batch.jl` stays (it was never exported); `four_index_integrals` is re-exported in Task 7.

- [ ] **Step 4: Verify the package loads** (tests will be red until Task 6 migrates them — that's expected):

Run: `julia --project=. -e 'using BoundaryIntegral; println("loads")'`
Expected: `loads` (no `UndefVarError` from dangling references). If a non-test file references a deleted symbol, fix it now.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete .bie parser + the SystemInput per-center API"
```

---

### Task 6: migrate the affected tests + fixtures to `.toml`; green suite

**Files:**
- Delete: `test/fixtures/system_*.bie`
- Create: `test/fixtures/system_smooth_lat.toml` (the one fixture still needed for `lattice_batch.jl`'s solve setup)
- Delete: `test/utils/system_input.jl` (replaced by `test/campaign/toml_input.jl` from Task 1)
- Modify: `test/solver/lattice_batch.jl`, `test/solver/multi_rhs.jl`, `test/runtests.jl`

- [ ] **Step 1: Create `test/fixtures/system_smooth_lat.toml`** (replaces `system_smooth_lat.bie`; orbital 2 = orb_smooth translated by one cell along a1. orb_smooth centroid is the blob center ≈ (1.0,1.0,1.0); the .bie used `LATTICE 1 0 0` = +one cell. Express that as an explicit second orbital at centroid + a1. Determine a1 from the template: `true_cell_vectors`/nx along axis 1; for orb_smooth, true cell 3.0 → a1 ≈ (3.0,0,0), so orbital 2 at centroid + (3,0,0).)

```toml
name = "system_smooth_lat"
root = "/tmp/system_smooth_lat_unused"
templates = ["orb_smooth.xsf"]

[[orbital]]
type = 1
x = 1.0
y = 1.0
z = 1.0

[[orbital]]
type = 1
x = 4.0
y = 1.0
z = 1.0

[pairing]
neighbor_cutoff = 5.0

[dielectrics]
eps_out = 1.0
boxes = [[1.5, 1.5, 1.5, 8.0, 8.0, 8.0, 2.0]]

[solve]
n_quad = 4
l_ec = 2.0
rhs_tol = 1e-2
lhs_tol = 1e-6
gmres_rtol = 1e-8
support_rtol = 1e-6
volume_tol = 1e-8
max_order = 8
max_depth = 128

[batching]
n_centers_per_batch = 1

[eval]
far_pad_steps = 2.0
```

(Verify the orbital centroid + a1 with a one-off: `julia --project=. -e 'using BoundaryIntegral; st,dg=read_xsf("test/fixtures/orb_smooth.xsf"); println(density_centroid(dg)); println(true_cell_vectors(dg))'` and set orbital-2 `x` = centroid.x + (true_cell_x). Adjust the literal if needed — the snap test in Task 2 confirms the step math.)

- [ ] **Step 2: Rewrite `test/solver/lattice_batch.jl`'s `.bie`-dependent testsets.** The file currently builds its solve reference via `read_system_input(system_smooth_lat.bie)` + `assemble_rhs_group` + `build_group_interface`. Replace the shared setup so it builds a `LatticeBatch` directly from the `.toml` and the templates, and the interface via `solve_dielectric_lattice_batch`:

Find the block (after the assemble tests) that does:
```julia
si = read_system_input(joinpath(fixdir, "system_smooth_lat.bie"))
g = assemble_rhs_group(si, 1; support_rtol = 1e-6)
interface = build_group_interface(si, g; n_quad = NQ, rhs_atol = RTOL_REFINE, l_ec = LEC)
```
Replace with:
```julia
c = load_campaign(joinpath(fixdir, "system_smooth_lat.toml"))
st, dg = BoundaryIntegral.read_xsf(c.templates[1])
insts = Dict(1 => OrbitalInstance(1, 1, (0,0,0)),
             2 => OrbitalInstance(2, 1, lattice_grid_steps(dg, st.primvec, (1,0,0))))
b = assemble_lattice_batch([dg], insts, [(1,1),(1,2)]; support_rtol = 1e-6)
res = solve_dielectric_lattice_batch(c.boxes, c.epses, c.eps_out, b;
    n_quad = NQ, rhs_atol = RTOL_REFINE, l_ec = LEC, fmm_tol = FT, gmres_rtol = GTOL)
interface = res.interface
```
Then DELETE the two testsets that asserted `assemble_lattice_batch == assemble_rhs_group` and `solve_dielectric_lattice_batch == group solve` (their reference — the old path — no longer exists; the remaining direct tests of `assemble_lattice_batch`, the TKM-everywhere check, the well-resolved near/far consistency, and `four_index_matrix` cover correctness). Keep the `four_index_matrix` comparison (now using `res`/`b` and explicit tolerances per Task 4). Repoint any `si_e`/`si_e.solve[...]` to `c`/`c.solve[...]` or literals.

- [ ] **Step 3: Rewrite `test/solver/multi_rhs.jl`.** It opens with `read_system_input` + `assemble_rhs_group` + `build_group_interface` to get a coarse interface, then tests the batched operator / batched RHS / block solve / `rhs_dielectric_box3d_fmm3d` agreement. Replace the interface construction with the `.toml`→`solve_dielectric_lattice_batch` setup (same pattern as Step 2, using `system_small.toml`), and DELETE the subtests that referenced `assemble_rhs_group`/`rhs_dielectric_box3d_fmm3d_batched`/`solve_dielectric_box3d_group`/`four_index_integrals(bie)` (the `.bie` per-center API). Keep the subtests that exercise the still-present multi-source core: `rhs_dielectric_box3d_fmm3d(interface, vss, thresh)`, `batched_lhs_dielectric_box3d_fmm3d_corrected`, `solve_dielectric_box3d_block`, the `mul!`/columnwise agreement. Build their `interface`/`vss` from the `.toml` batch.

(Concretely: replace the header block with
```julia
c = load_campaign(joinpath(fixdir, "system_small.toml"))
st, dg1 = BoundaryIntegral.read_xsf(c.templates[1])
insts = Dict(1 => OrbitalInstance(1, 1, (0,0,0)))
b = assemble_lattice_batch([dg1], insts, [(1,1)]; support_rtol = 1e-6)
res = solve_dielectric_lattice_batch(c.boxes, c.epses, c.eps_out, b;
    n_quad = NQ, rhs_atol = RTOL_REFINE, l_ec = LEC, fmm_tol = FT, gmres_rtol = GTOL)
interface = res.interface
vss = BoundaryIntegral.batch_volume_sources(b)
```
and adapt the remaining assertions to use `interface`/`vss`. The exact final file is produced by the implementer reading the current testset and keeping only the non-`.bie` assertions.)

- [ ] **Step 4: Delete the old fixtures and parser test, update runtests**

```bash
git rm test/fixtures/system_small.bie test/fixtures/system_lat.bie test/fixtures/system_spike.bie test/fixtures/system_centroid.bie test/fixtures/system_pair.bie test/fixtures/system_smooth_lat.bie
git rm test/utils/system_input.jl
```
In `test/runtests.jl`, remove `include("utils/system_input.jl")` (replaced by `include("campaign/toml_input.jl")` added in Task 1).

- [ ] **Step 5: Run the full suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS. Fix any remaining dangling `.bie`/`SystemInput`/`assemble_rhs_group` references the migration missed.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test: migrate .bie fixtures/tests to .toml; drop old-path equivalence tests"
```

---

## PHASE B — fold the campaign phases into core + in-memory `four_index_integrals`

### Task 7: campaign tasks (serial) + in-memory `four_index_integrals(toml)`

**Files:**
- Create: `src/campaign/tasks.jl` (adapted from CampaignLib's `tasks.jl`)
- Modify: `src/BoundaryIntegral.jl` (include after `manifest.jl`; `using Serialization` already present; exports)
- Create: `test/campaign/pipeline.jl` (adapted mini-campaign pipeline + in-memory test)
- Create: `test/campaign/fixture_campaign.jl` (the tiny generated campaign, `.toml` form)
- Modify: `test/runtests.jl`

- [ ] **Step 1: Create `test/campaign/fixture_campaign.jl`** — a helper that writes a tiny self-contained `.toml` campaign (the orb_smooth blob, 2 orbitals one cell apart, into a tmp root). Adapt CampaignLib's `test/fixture_campaign.jl`: write the xsf as before, but emit the new `.toml` (explicit `[[orbital]]` ×2, `templates=[orb.xsf]`, `[pairing] neighbor_cutoff`, `[dielectrics]`, `[solve]`, `[batching]`, `[eval]`).

- [ ] **Step 2: Write the failing tests `test/campaign/pipeline.jl`**

```julia
using BoundaryIntegral, Serialization, LinearAlgebra, Test
include("fixture_campaign.jl")

@testset "prepare + serial phases + in-memory" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c)
        @test isfile(manifest_path(c)) && isfile(centers_path(c))
        @test sort(pending_batches(c, :solve)) == [1, 2]
        for id in [1, 2]; solve_batch(c, id); end
        @test isempty(pending_batches(c, :solve))
        consolidate(c)
        for id in [1, 2]; eval_batch(c, id); end
        rep = assemble_v(c)
        @test rep.max_rel_asym < 1e-2
        @test isfile(joinpath(c.root, "V_full.tsv"))

        # in-memory path: same .toml, no files, must match the batched V
        res = four_index_integrals(write_fixture_campaign(mktempdir()))
        full = open(deserialize, joinpath(c.root, "V_full.jls"))   # written alongside tsv (see Task 8)
        @test res.pair_ids == full.pair_ids
        @test maximum(abs.(res.V .- full.V)) < 1e-8 * max(maximum(abs.(full.V)), eps())
    end
end
```

(NOTE: this test references `V_full.jls`; Task 8 changes `assemble_v` to write `.tsv` and DROP `.jls`. Adjust the in-memory cross-check in Task 8 to compare `res.V` against the parsed `.tsv` instead. For Task 7, compare `res.V` to a freshly-assembled in-memory `V` from the same campaign rather than to a file.)

Revised Task-7 in-memory assertion (no file dependency):
```julia
        res = four_index_integrals(c)            # accepts a CampaignInput too
        # rebuild dense V from the per-batch V files for comparison
        store = open(deserialize, rho_store_path(c)); pid = store.pair_ids
        col = Dict(p => i for (i, p) in enumerate(pid)); Vb = fill(NaN, length(pid), length(pid))
        for b in read_manifest(manifest_path(c))
            vr = BoundaryIntegral.load_v_rows(v_path(c, b.batch_id))
            for (k, sp) in enumerate(vr.source_pairs); Vb[:, col[sp]] = vr.V[:, k]; end
        end
        @test res.pair_ids == pid
        @test maximum(abs.(res.V .- Vb)) < 1e-8 * max(maximum(abs.(Vb)), eps())
```

- [ ] **Step 3: Run, verify failure** → `prepare not defined`.

- [ ] **Step 4: Implement `src/campaign/tasks.jl`.** Port CampaignLib's `tasks.jl` with these changes:
  - `Campaign` → `CampaignInput` everywhere; `c.xsf` → `c.templates`; `campaign_l_ec(c)` already defined in `toml_input.jl`.
  - `prepare(c)`: replace the `enumerate_centers(nx, ny, ...)` block with `centers = enumerate_centers(c)` (Task 3). Keep the `manifest.params` drift guard, but `_params_string(c)` now hashes `(length(c.orbitals), c.neighbor_cutoff, c.n_centers_per_batch, c.pair_overrides)`. If `c.pair_overrides !== nothing`, use it directly instead of `enumerate_pairs`.
  - `solve_batch`, `consolidate`, `eval_batch`, `pending_batches`, `_batch_instances`, `save_v_rows`/`load_v_rows`, `_is_complete_v`, `_atomic_serialize` — port verbatim (they already use `BatchResult`/`OrbitalInstance`/`evaluate_batch_potential` from core; `gethostname` needs `using Sockets`).
  - Add the **`*_core`** refactor so the in-memory path reuses logic:
    - `solve_batch_core(c, spec, grids, insts) -> BatchResult` (assemble + solve + pack, no file). `solve_batch` = `is_complete_batch` check → `solve_batch_core` → `save_batch_result`.
    - `consolidate_core(brs::Vector{BatchResult}, dg) -> (targets, store)`; `consolidate` = load brs → core → serialize.
    - `eval_batch_core(br, targets, store, dg, c) -> (source_pairs, V)`; `eval_batch` = load → core → `save_v_rows`.
  - **`four_index_integrals(c::CampaignInput)`** (and a `::AbstractString` method that does `load_campaign` first): run all `*_core` in-process, assemble the dense `V`, return `(; pair_ids, V)`. No files written.

```julia
using Sockets

function four_index_integrals(c::CampaignInput)
    centers = enumerate_centers(c)
    byid = Dict(ct.id => ct for ct in centers)
    pairs = c.pair_overrides === nothing ? enumerate_pairs(centers, c.neighbor_cutoff) : c.pair_overrides
    batches = build_batches(pairs, c.n_centers_per_batch)
    temps = load_templates!(c); grids = [t[2] for t in temps]; dg = grids[1]
    insts(spec) = Dict(id => OrbitalInstance(id, byid[id].template_id, byid[id].steps)
                       for id in unique(reduce(vcat, [[p[1],p[2]] for p in spec.pairs]; init=Int[])))
    brs = [solve_batch_core(c, spec, grids, insts(spec)) for spec in batches]
    targets, store = consolidate_core(brs, dg, c)
    pid = store.pair_ids; col = Dict(p => i for (i, p) in enumerate(pid))
    V = fill(NaN, length(pid), length(pid))
    for br in brs
        sp_pairs, Vb = eval_batch_core(br, targets, store, dg, c)
        for (k, sp) in enumerate(sp_pairs); V[:, col[sp]] = Vb[:, k]; end
    end
    return (; pair_ids = pid, V)
end
four_index_integrals(toml_path::AbstractString) = four_index_integrals(load_campaign(toml_path))
```

In `src/BoundaryIntegral.jl`: `include("campaign/tasks.jl")` after `manifest.jl`; export `prepare, solve_batch, consolidate, eval_batch, assemble_v, pending_batches, four_index_integrals`. Add `test/campaign/pipeline.jl` to `test/runtests.jl`. Add `Sockets` to `Project.toml` `[deps]`+`[compat]` if not present.

- [ ] **Step 5: Run, verify pass**

Run: `julia --project=. test/campaign/pipeline.jl` → PASS (a couple minutes — real coarse solves).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: fold campaign phases into BI.jl core + in-memory four_index_integrals(toml)"
```

---

## PHASE C — text V output

### Task 8: `V_full.tsv` text output

**Files:**
- Create: `src/campaign/v_output.jl` (`write_v_table`)
- Modify: `src/campaign/tasks.jl` (`assemble_v` writes `.tsv`, drops `.jls`)
- Modify: `src/BoundaryIntegral.jl` (include; export `write_v_table`)
- Test: `test/campaign/v_output.jl` (create) + adjust `test/campaign/pipeline.jl`

- [ ] **Step 1: Write the failing test `test/campaign/v_output.jl`**

```julia
using BoundaryIntegral, Test

@testset "v_output round-trip" begin
    pair_ids = [(1,1),(1,2),(2,2)]
    V = [1.0 2.0 3.0; 2.0 4.0 5.0; 3.0 5.0 6.0]
    mktempdir() do dir
        p = joinpath(dir, "V_full.tsv")
        BoundaryIntegral.write_v_table(p, pair_ids, V)
        lines = readlines(p)
        @test lines[1] == "i\tj\tk\tl\tV"
        @test length(lines) == 1 + 9          # header + 3×3 entries
        # parse back
        got = Dict{NTuple{4,Int},Float64}()
        for ln in lines[2:end]
            f = split(ln, '\t'); got[(parse(Int,f[1]),parse(Int,f[2]),parse(Int,f[3]),parse(Int,f[4]))] = parse(Float64, f[5])
        end
        @test got[(1,1,1,2)] == 2.0    # (i,j)=pair_ids[1]=(1,1), (k,l)=pair_ids[2]=(1,2)
        @test got[(2,2,2,2)] == 6.0
        @test length(got) == 9
    end
end
```

- [ ] **Step 2: Run, verify failure** → `write_v_table not defined`.

- [ ] **Step 3: Implement `src/campaign/v_output.jl`**

```julia
"""
    write_v_table(path, pair_ids, V)

Write the four-index result as a text table: header `i\\tj\\tk\\tl\\tV`, then one row per
entry `i j k l value` where `(i,j) = pair_ids[row]`, `(k,l) = pair_ids[col]`,
`value = V[row, col]`. All entries (dense). Atomic (tmp + rename).
"""
function write_v_table(path::AbstractString, pair_ids::Vector{Tuple{Int,Int}}, V::AbstractMatrix)
    nr, nc = size(V)
    (nr == length(pair_ids) && nc == length(pair_ids)) ||
        throw(DimensionMismatch("V is $(size(V)) but pair_ids has $(length(pair_ids))"))
    d = dirname(path); isempty(d) || mkpath(d)
    tmp = string(path, ".tmp.", getpid())
    open(tmp, "w") do io
        println(io, "i\tj\tk\tl\tV")
        for r in 1:nr, cc in 1:nc
            (i, j) = pair_ids[r]; (k, l) = pair_ids[cc]
            println(io, i, '\t', j, '\t', k, '\t', l, '\t', V[r, cc])
        end
    end
    mv(tmp, path; force = true)
    return path
end
```

In `src/BoundaryIntegral.jl`: `include("campaign/v_output.jl")` after `tasks.jl` (or before — it has no deps); export `write_v_table`.

- [ ] **Step 4: Change `assemble_v`** in `src/campaign/tasks.jl` to write `V_full.tsv` and stop writing `V_full.jls`: replace the `_atomic_serialize(joinpath(c.root, "V_full.jls"), (; pair_ids, V))` line with
```julia
    write_v_table(joinpath(c.root, "V_full.tsv"), pair_ids, V)
```
Keep `report.txt`. Update the `test/campaign/pipeline.jl` assertion from `isfile(... "V_full.jls")` to `isfile(... "V_full.tsv")`, and (from Task 7's note) drop any `V_full.jls` deserialize — the in-memory cross-check already compares against the per-batch V files.

- [ ] **Step 5: Run** `julia --project=. test/campaign/v_output.jl` and `julia --project=. test/campaign/pipeline.jl` → PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: write V_full.tsv text table (i j k l V), drop V_full.jls"
```

---

## PHASE D — Distributed/SlurmClusterManager extension

### Task 9: `BoundaryIntegralDistributedExt`

**Files:**
- Create: `ext/BoundaryIntegralDistributedExt.jl`
- Modify: `Project.toml` (`[weakdeps]` += Distributed, SlurmClusterManager; `[extensions]` += the ext; `[compat]`)
- Modify: `src/BoundaryIntegral.jl` (declare the stub function `run_phase` that the ext implements)
- Test: `test/campaign/distributed_ext.jl` (create; local-workers path only)

- [ ] **Step 1: Declare the extension entry point** in core. In `src/campaign/tasks.jl` add a stub:
```julia
"""
    run_phase(c, phase; workers=0)

Parallel driver for `:solve`/`:eval` over `pending_batches`. Implemented by the
`BoundaryIntegralDistributedExt` extension (load `Distributed` + `SlurmClusterManager`).
Without the extension loaded, this errors with a hint.
"""
function run_phase end
```
Export `run_phase` in `src/BoundaryIntegral.jl`.

- [ ] **Step 2: Add weakdeps/extension to `Project.toml`**

```bash
julia --project=. -e 'using Pkg; Pkg.add(["Distributed","SlurmClusterManager"])'   # adds to [deps]
```
Then MOVE `Distributed` and `SlurmClusterManager` from `[deps]` to `[weakdeps]` (next to `Makie`), and add under `[extensions]`:
```toml
BoundaryIntegralDistributedExt = ["Distributed", "SlurmClusterManager"]
```
Add `[compat]` entries (`Distributed = "1.11"`, `SlurmClusterManager = "1"` — check the installed version with `Pkg.status` and pin accordingly).

- [ ] **Step 3: Write the failing test `test/campaign/distributed_ext.jl`** (local workers, no Slurm)

```julia
using BoundaryIntegral, Distributed, SlurmClusterManager, Test
include("fixture_campaign.jl")

@testset "run_phase local workers" begin
    mktempdir() do dir
        c = load_campaign(write_fixture_campaign(dir))
        prepare(c)
        run_phase(c, :solve; workers = 2)         # local addprocs(2)
        @test isempty(pending_batches(c, :solve))
        consolidate(c)
        run_phase(c, :eval; workers = 2)
        @test isempty(pending_batches(c, :eval))
        rmprocs(workers())                         # clean up
    end
end
```

- [ ] **Step 4: Run, verify failure** → `run_phase` errors (no extension method) or MethodError.

- [ ] **Step 5: Implement `ext/BoundaryIntegralDistributedExt.jl`**

```julia
module BoundaryIntegralDistributedExt

using BoundaryIntegral
using Distributed
using SlurmClusterManager

# Spawn workers: SlurmManager() inside a Slurm allocation (one worker/task, reads env),
# local addprocs(workers) otherwise, or none (inline) when workers==0 and not on Slurm.
function _spawn(workers::Int)
    proj = dirname(Base.active_project())
    glue = get(ENV, "JULIA_GLUE_THREADS", "8")
    if haskey(ENV, "SLURM_JOB_ID") && parse(Int, get(ENV, "SLURM_NTASKS", "1")) > 1
        @info "run_phase: SlurmManager workers"
        addprocs(SlurmManager(); exeflags = `--project=$proj -t $glue`)
    elseif workers > 0
        @info "run_phase: $workers local workers"
        addprocs(workers; exeflags = `--project=$proj -t $glue`)
    end
end

function BoundaryIntegral.run_phase(c::BoundaryIntegral.CampaignInput, phase::Symbol; workers::Int = 0)
    runner = phase === :solve ? BoundaryIntegral.solve_batch :
             phase === :eval  ? BoundaryIntegral.eval_batch  :
             error("run_phase: phase must be :solve or :eval")
    _spawn(workers)
    @everywhere eval(:(using BoundaryIntegral))
    pending = pending_batches(c, phase)
    toml = c.toml_path                         # workers reload from the .toml path (cheap, cached)
    results = pmap(pending; retry_delays = [30.0], on_error = e -> e) do id
        try
            runner(BoundaryIntegral.load_campaign(toml), id)
            (id, :ok)
        catch err
            cc = BoundaryIntegral.load_campaign(toml)
            mkpath(logs_dir(cc))
            write(joinpath(logs_dir(cc), "$(phase)_batch_$(lpad(id, 4, '0')).err"),
                  sprint(showerror, err, catch_backtrace()))
            rethrow()
        end
    end
    ok = count(r -> r isa Tuple && r[2] === :ok, results)
    @info "run_phase finished" phase ok failed=length(results)-ok
    return results
end

end
```

`run_phase` uses `c.toml_path` (the field added to `CampaignInput` in Task 1) so workers reload the campaign from disk; the `_spawn`/`@everywhere`/`pmap`/error-log structure mirrors the old `driver.jl` (Task 14 of the prior plan).

- [ ] **Step 6: Run, verify pass**

Run: `julia --project=. test/campaign/distributed_ext.jl` → PASS. (Add to `test/runtests.jl` ONLY if `Distributed`/`SlurmClusterManager` are in the test env — add them to `[extras]`/`test` target; the local-workers test is the inline path, fast.)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: BoundaryIntegralDistributedExt (run_phase via SlurmClusterManager); CampaignInput.toml_path"
```

---

## PHASE E — work-repo deployment layer

### Task 10: thin driver + sbatch + campaign `.toml`s; remove standalone CampaignLib; re-validate

**Repo:** `/mnt/home/xgao1/work/four_index_integral_solver` (commits on `main`)

**Files:**
- Rewrite: `codes/lattice_scale/driver.jl` (thin: load BI.jl + extension, dispatch phases / `run_phase`)
- Modify: `codes/lattice_scale/jobscripts/*.sbatch` (julia loads the extension; otherwise unchanged)
- Convert: `codes/lattice_scale/campaigns/*.toml` to the new schema (`[[orbital]]` lists; the 10×10 generated explicitly)
- Remove: `codes/lattice_scale/src/` (the standalone CampaignLib), `Project.toml` CampaignLib entry → `Project.toml` now just `dev`s BoundaryIntegral + adds Distributed/SlurmClusterManager
- Rewrite: `codes/lattice_scale/scripts/compare_anchor.jl` (in-memory `four_index_integrals(toml)` vs batched `V_full.tsv`)

- [ ] **Step 1: Update the work-repo Project.toml** — remove the `CampaignLib` package definition; make it a plain environment that `dev`s BoundaryIntegral and depends on `Distributed`, `SlurmClusterManager`. `git rm -r codes/lattice_scale/src` and the CampaignLib `Project.toml`/`test`. Re-instantiate:
```bash
cd /mnt/home/xgao1/work/four_index_integral_solver/codes/lattice_scale
julia --project=. -e 'using Pkg; Pkg.develop(path="/mnt/home/xgao1/codes/BoundaryIntegral.jl"); Pkg.add(["Distributed","SlurmClusterManager"]); Pkg.instantiate()'
```

- [ ] **Step 2: Rewrite `driver.jl`** to a thin wrapper:
```julia
using BoundaryIntegral, Distributed, SlurmClusterManager
toml, phase = ARGS[1], Symbol(ARGS[2])
only_id = length(ARGS) >= 4 && ARGS[3] == "--only" ? parse(Int, ARGS[4]) : nothing
workers = length(ARGS) >= 4 && ARGS[3] == "--workers" ? parse(Int, ARGS[4]) : 0
c = load_campaign(toml)
if phase === :prepare; prepare(c)
elseif phase === :consolidate; consolidate(c)
elseif phase === :assemble; assemble_v(c)
elseif phase === :status; for ph in (:solve,:eval); println(ph, ": ", length(pending_batches(c, ph)), " pending"); end
elseif phase in (:solve, :eval)
    if only_id !== nothing
        (phase === :solve ? solve_batch : eval_batch)(c, only_id)
    else
        run_phase(c, phase; workers = workers)
    end
else error("unknown phase $phase") end
```

- [ ] **Step 3: Convert the campaign `.toml`s.** `demo_2x2.toml` → new schema with 8 explicit `[[orbital]]` entries (compute their positions: sublattice centroids + Rx·a1 + Ry·a2 for Rx,Ry∈{0,1}; reuse the values from the old `centers.tsv`). For `lattice_10x10.toml`, generate the 200 `[[orbital]]` entries with a one-off script (centroids + lattice sweep) written into the toml. Keep dielectrics/solve/batching/eval blocks (already toml).

- [ ] **Step 4: Rewrite `compare_anchor.jl`** to: run `four_index_integrals(toml)` in-memory for the small anchor, and compare against the batched `V_full.tsv` (parse the text), reporting max rel diff. (No more `.bie` reference — the two independent paths are in-memory-all-at-once vs batched-on-disk.)

- [ ] **Step 5: Re-validate** — hand the user the commands (they run Slurm): the inline 2×2 (`prepare` + `solve --only`/`--workers` + consolidate + eval + assemble), confirm `V_full.tsv` + `report.txt` symmetry ≈ 9e-6, and the 2-node `run_campaign.sbatch` smoke test on a fresh root. The implementer runs only the non-Slurm pieces (prepare/consolidate/assemble, `four_index_integrals` in-memory) to confirm they work; Slurm submission is the user's.

- [ ] **Step 6: Commit (work repo)**

```bash
cd /mnt/home/xgao1/work/four_index_integral_solver
git add -A codes/lattice_scale
git commit -m "lattice_scale: thin driver on BI.jl campaign API; .toml schema; remove standalone CampaignLib"
```

---

## Self-review checklist (run after writing; fix inline)

1. **Spec coverage:** §2 architecture → Tasks 1,3,7,9 (core) + 9 (ext) + 10 (deployment); §3 .toml format → Task 1; §4 snap geometry → Task 2; §5 .bie removal + in-memory API + fixtures → Tasks 4,5,6,7; §6 V text output → Task 8; §7 SlurmClusterManager → Task 9; §8 testing → tests in every task; §9 sequencing → Phases A–E map 1:1. Covered.
2. **No placeholders:** new code is complete; the two test-migration tasks (6 Step 3, 10 Step 3/4) give the exact setup-block replacement + the keep/delete rule because the full rewritten file depends on reading the current testset — the implementer reads the current file and applies the stated transformation (this is a migration, not greenfield; the transformation is fully specified).
3. **Type consistency:** `CampaignInput` (incl. the `toml_path` field, present from Task 1) is used consistently; `CenterInfo(id, template_id, steps, center)` (4 fields, no Rx/Ry) matches across Tasks 3/7; `four_index_matrix(interface, sources, Σ; lhs_tol, volume_tol)` (Task 4) matches its caller in Task 6; `four_index_integrals(::CampaignInput)`/`(::AbstractString)` (Task 7) matches the in-memory test; `write_v_table(path, pair_ids, V)` (Task 8) matches `assemble_v`'s call; `run_phase(c, phase; workers)` (Task 9) matches the driver.
