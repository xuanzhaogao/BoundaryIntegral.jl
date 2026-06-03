# Per-Center Multi-RHS Batched BIE Solve — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Solve the dielectric BIE for a whole per-center group of pair densities $\{\rho_{ij}=\varphi_i\varphi_j\}_{j\in\mathrm{neigh}(i)}$ in one shared-operator block — driven by a Step-0 `.bie` system file — using a vectorized (`nd`-batched) FMM matvec and block GMRES, stopping at the layer densities $\Sigma$.

**Architecture:** A `.bie` text file (Unit 0) describes the dielectric geometry + orbital `.xsf` paths + cutoff/override grouping. For each center it loads the orbitals, forms pair densities by pointwise product on a shared grid (Unit 1), builds one union/envelope-refined interface (Unit 2), assembles the $N\times K$ RHS matrix with one batched FMM (Unit 4), and applies a custom matrix-capable operator (Unit 5) inside `Krylov.block_gmres` (Unit 6).

**Tech Stack:** Julia, FMM3D (`lfmm3d`, batched via `nd`), Krylov.jl (`block_gmres`), LinearAlgebra/SparseArrays, existing `BoundaryIntegral` panel/source/solver machinery.

**Reference spec:** `docs/superpowers/specs/2026-06-03-per-center-multi-rhs-design.md`

**Key conventions discovered (do not re-derive):**
- `lfmm3d(eps, sources; charges, targets, pg, pgt, nd)`: `charges` is `(nd, n)`, `vals.grad` is `(nd, 3, n)`, `vals.gradtarg` is `(nd, 3, nt)`. **The leading axis is the density index.**
- Single-RHS FMM rhs convention (`rhs_dielectric_box3d_fmm3d`, `src/solver/dielectric_box3d.jl:127`): with screened density and `eps_src=1.0`, `Rhs[i] = dot(normals[:,i], gradtarg[:,i]) / (4π * eps_src)`. The K=1 regression test pins all sign/screening conventions — mirror this exactly.
- `screened_volume_source(boxes, epses, eps_out, vs, SharpScreening())` scales each density value by `1/eps_local(position)` and **preserves positions/weights** (`src/core/sources.jl:343,361`). So a group of densities on one shared grid stays on shared positions after screening → `nd`-batchable.
- Operator pieces: `AdaptiveConfig(atol, rtol, n_GL, max_depth)`; `build_neighbor_list(interface, max_order, up_tol; range_factor, correct_edges, adaptive_cfg) -> (; upsample, adaptive)`; `laplace3d_DT_corrections(interface, upsample, adaptive) -> sparse matrix` (all in `src/kernel/laplace3d_near.jl`).
- Run a single test file: `julia --project -e 'using Pkg; Pkg.test()'` runs the default set; to run one file in isolation use `julia --project test/runtests.jl` after adding the include, or `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`.

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `src/utils/system_input.jl` | `.bie` parser, `SystemInput`/`DielectricBox`/`OrbitalEntry` types, group resolution | Create |
| `src/utils/xsf_reader.jl` | add `VolumeSource(datagrid, density; ...)` overload + `datagrids_compatible` | Modify |
| `src/solver/multi_rhs.jl` | `RHSGroup`, `pair_density_source`, `assemble_rhs_group`, envelope source, `BatchedDielectricOperator` + `mul!`, `solve_dielectric_box3d_group` | Create |
| `src/solver/dielectric_box3d.jl` | add `rhs_dielectric_box3d_fmm3d_batched(interface, group, …)` | Modify |
| `src/BoundaryIntegral.jl` | `include`s + `export`s for the new symbols | Modify |
| `test/utils/system_input.jl` | parser tests + fixtures | Create |
| `test/solver/multi_rhs.jl` | pair-density, group, batched-RHS, matvec, block-solve, integration tests | Create |
| `test/fixtures/` | small `.bie` + `.xsf` fixtures | Create |
| `test/runtests.jl` | register new test files | Modify |

---

## Task 1: `SystemInput` types and `.bie` parser (Unit 0)

**Files:**
- Create: `src/utils/system_input.jl`
- Modify: `src/BoundaryIntegral.jl` (include + exports)
- Create: `test/fixtures/system_small.bie`
- Create: `test/utils/system_input.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Create the fixture `.bie` and two tiny `.xsf` files**

Create `test/fixtures/system_small.bie`:

```
# small system fixture
UNITS bohr

BEGIN_DIELECTRICS
EPS_OUT 1.0
  0.0 0.0 0.0    4.0 4.0 4.0    11.7
END_DIELECTRICS

BEGIN_ORBITALS
  1   orb_a.xsf
  2   orb_b.xsf
END_ORBITALS

BEGIN_GROUPING
CUTOFF 3.0
  2 : 2
END_GROUPING
```

Create `test/fixtures/orb_a.xsf` (single atom at origin, 2×2×2 grid of ones):

```
CRYSTAL
PRIMVEC
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
PRIMCOORD
1 1
X 0.0 0.0 0.0
BEGIN_BLOCK_DATAGRID_3D
 density
BEGIN_DATAGRID_3D_density
2 2 2
0.0 0.0 0.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
1.0 1.0 1.0 1.0
1.0 1.0 1.0 1.0
END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
```

Create `test/fixtures/orb_b.xsf`: identical but `PRIMCOORD` atom at `X 5.0 0.0 0.0` (so its center is 5.0 away from orb_a → outside the 3.0 cutoff), same datagrid.

- [ ] **Step 2: Write the failing parser test**

Create `test/utils/system_input.jl`:

```julia
using BoundaryIntegral
using Test

@testset "system_input parser" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")
    si = read_system_input(joinpath(fixdir, "system_small.bie"))

    # units
    @test si.unit_scale == 1.0

    # dielectrics
    @test length(si.boxes) == 1
    @test si.boxes[1].center == (0.0, 0.0, 0.0)
    @test si.boxes[1].Lx == 4.0
    @test si.epses == [11.7]
    @test si.eps_out == 1.0

    # orbitals: centers read from PRIMCOORD
    @test si.orbitals[1].center == (0.0, 0.0, 0.0)
    @test si.orbitals[2].center == (5.0, 0.0, 0.0)
    @test endswith(si.orbitals[1].xsf_path, "orb_a.xsf")

    # grouping: cutoff 3.0 -> orbital 1 groups only with itself (orb 2 is 5.0 away);
    # explicit override sets group of 2 to [2]
    @test sort(si.groups[1]) == [1]
    @test sort(si.groups[2]) == [2]

    # malformed: multi-atom PRIMCOORD without atom_index should error
    @test_throws ErrorException read_system_input(
        joinpath(fixdir, "system_multiatom_bad.bie"))
end
```

Also create `test/fixtures/system_multiatom_bad.bie` (same as `system_small.bie` but pointing orbital 1 at a new `orb_multi.xsf` whose `PRIMCOORD` lists `2 1` with two atoms and **no** `atom_index` column in the orbital row) plus `test/fixtures/orb_multi.xsf` (two atoms in PRIMCOORD, otherwise like `orb_a.xsf`).

- [ ] **Step 3: Register the test and run it to confirm failure**

In `test/runtests.jl`, under the `# utilities` group add:

```julia
    include("utils/system_input.jl")
```

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/utils/system_input.jl")'`
Expected: FAIL — `read_system_input` not defined.

- [ ] **Step 4: Implement the parser**

Create `src/utils/system_input.jl`:

```julia
const BoxGeom = NamedTuple{(:center, :Lx, :Ly, :Lz),
    Tuple{NTuple{3,Float64}, Float64, Float64, Float64}}

struct OrbitalEntry
    id::Int
    xsf_path::String
    center::NTuple{3,Float64}
end

struct SystemInput
    unit_scale::Float64
    boxes::Vector{BoxGeom}
    epses::Vector{Float64}
    eps_out::Float64
    orbitals::Dict{Int,OrbitalEntry}
    groups::Dict{Int,Vector{Int}}
end

_clean(line) = strip(first(split(line, '#')))   # drop comments

function read_system_input(path::AbstractString)
    base = dirname(abspath(path))
    raw = readlines(path)
    lines = [(_clean(l)) for l in raw]

    unit_scale = 1.0
    boxes = BoxGeom[]
    epses = Float64[]
    eps_out = 1.0
    # orbital raw rows: (id, xsf_path, atom_index_or_0)
    orb_rows = Tuple{Int,String,Int}[]
    cutoff = Inf
    overrides = Dict{Int,Vector{Int}}()

    i = 1
    while i <= length(lines)
        s = lines[i]
        if isempty(s)
            i += 1; continue
        elseif startswith(s, "UNITS")
            u = lowercase(split(s)[2])
            unit_scale = u == "bohr" ? 1.0 :
                         u == "angstrom" ? 1.8897259886 :
                         error("Unknown UNITS: $u")
            i += 1
        elseif s == "BEGIN_DIELECTRICS"
            i += 1
            while i <= length(lines) && lines[i] != "END_DIELECTRICS"
                row = lines[i]
                if isempty(row)
                    i += 1; continue
                elseif startswith(row, "EPS_OUT")
                    eps_out = parse(Float64, split(row)[2])
                else
                    p = parse.(Float64, split(row))
                    length(p) == 7 || error("DIELECTRIC row needs 7 numbers, got: $row")
                    push!(boxes, (center = (p[1], p[2], p[3]),
                                  Lx = p[4], Ly = p[5], Lz = p[6]))
                    push!(epses, p[7])
                end
                i += 1
            end
            i += 1  # skip END
        elseif s == "BEGIN_ORBITALS"
            i += 1
            while i <= length(lines) && lines[i] != "END_ORBITALS"
                row = lines[i]
                if isempty(row); i += 1; continue; end
                parts = split(row)
                id = parse(Int, parts[1])
                xsf = parts[2]
                atom_idx = length(parts) >= 3 ? parse(Int, parts[3]) : 0
                push!(orb_rows, (id, xsf, atom_idx))
                i += 1
            end
            i += 1
        elseif s == "BEGIN_GROUPING"
            i += 1
            while i <= length(lines) && lines[i] != "END_GROUPING"
                row = lines[i]
                if isempty(row); i += 1; continue; end
                if startswith(row, "CUTOFF")
                    cutoff = parse(Float64, split(row)[2])
                elseif occursin(':', row)
                    lhs, rhs = split(row, ':')
                    ci = parse(Int, strip(lhs))
                    overrides[ci] = parse.(Int, split(strip(rhs)))
                else
                    error("Unrecognized GROUPING line: $row")
                end
                i += 1
            end
            i += 1
        else
            error("Unrecognized top-level line: $s")
        end
    end

    # resolve orbital centers from each .xsf PRIMCOORD
    orbitals = Dict{Int,OrbitalEntry}()
    for (id, xsf, atom_idx) in orb_rows
        full = isabspath(xsf) ? xsf : joinpath(base, xsf)
        structure, _ = read_xsf(full)
        nat = size(structure.positions, 1)
        a = atom_idx == 0 ? (nat == 1 ? 1 :
            error("orbital $id: PRIMCOORD has $nat atoms; specify atom_index")) :
            atom_idx
        (1 <= a <= nat) || error("orbital $id: atom_index $a out of range 1:$nat")
        c = ntuple(d -> structure.positions[a, d] * unit_scale, 3)
        orbitals[id] = OrbitalEntry(id, full, c)
    end

    # scale geometry by units
    boxes = BoxGeom[(center = b.center .* unit_scale,
                     Lx = b.Lx * unit_scale, Ly = b.Ly * unit_scale,
                     Lz = b.Lz * unit_scale) for b in boxes]
    cutoff_scaled = cutoff * unit_scale

    # resolve groups: cutoff then overrides
    ids = sort(collect(keys(orbitals)))
    groups = Dict{Int,Vector{Int}}()
    for i_id in ids
        ci = orbitals[i_id].center
        neigh = Int[]
        for j_id in ids
            cj = orbitals[j_id].center
            d = sqrt(sum(abs2, ci .- cj))
            d <= cutoff_scaled && push!(neigh, j_id)
        end
        groups[i_id] = neigh
    end
    for (ci, lst) in overrides
        groups[ci] = copy(lst)
    end

    return SystemInput(unit_scale, boxes, epses, eps_out, orbitals, groups)
end
```

Note: `UNITS` is parsed before geometry in the fixture; the parser collects `unit_scale` first then applies it after the full read, so ordering within the file is flexible.

- [ ] **Step 5: Wire into the module**

In `src/BoundaryIntegral.jl`, add after line 89 (`include("utils/xsf_reader.jl")`):

```julia
include("utils/system_input.jl")
```

And add an export near the other utility exports (after line 21):

```julia
export read_system_input, SystemInput, OrbitalEntry
```

- [ ] **Step 6: Run the test to confirm it passes**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/utils/system_input.jl")'`
Expected: PASS (all assertions, including the multi-atom `@test_throws`).

- [ ] **Step 7: Commit**

```bash
git add src/utils/system_input.jl src/BoundaryIntegral.jl test/utils/system_input.jl test/fixtures/ test/runtests.jl
git commit -m "feat: .bie system input parser (Step 0, Unit 0)"
```

---

## Task 2: Pair-density formation `rho_ij = phi_i * phi_j` (Unit 1a)

**Files:**
- Modify: `src/utils/xsf_reader.jl` (add density-override `VolumeSource` overload + `datagrids_compatible`)
- Create: `src/solver/multi_rhs.jl` (start it; add `pair_density_source`)
- Modify: `src/BoundaryIntegral.jl` (include + export)
- Create: `test/solver/multi_rhs.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write the failing test**

Create `test/solver/multi_rhs.jl`:

```julia
using BoundaryIntegral
using Test

@testset "multi_rhs" begin
    fixdir = joinpath(@__DIR__, "..", "fixtures")

    @testset "pair_density_source shared grid" begin
        # orb_a has all-ones density on a 2x2x2 grid. rho = phi_a * phi_a = 1 everywhere.
        _, dg = BoundaryIntegral.read_xsf(joinpath(fixdir, "orb_a.xsf"))
        vs = BoundaryIntegral.pair_density_source(
            joinpath(fixdir, "orb_a.xsf"), joinpath(fixdir, "orb_a.xsf"))
        @test length(vs.density) == 8          # 2*2*2 grid points
        @test all(vs.density .≈ 1.0)            # 1 * 1
        @test sum(vs.weights) ≈ 8.0             # cell volume 1.0 each (jacobian/(n))
    end
end
```

- [ ] **Step 2: Run it to confirm failure**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: FAIL — `pair_density_source` not defined.

- [ ] **Step 3: Add the `VolumeSource`-with-density overload and a grid-compat check to `xsf_reader.jl`**

In `src/utils/xsf_reader.jl`, after the existing `VolumeSource(datagrid; …)` (ends at line 162), add:

```julia
function datagrids_compatible(a, b; tol = 1e-10)
    a.nx == b.nx && a.ny == b.ny && a.nz == b.nz || return false
    maximum(abs.(a.origin .- b.origin)) <= tol &&
        maximum(abs.(a.A .- b.A)) <= tol &&
        maximum(abs.(a.B .- b.B)) <= tol &&
        maximum(abs.(a.C .- b.C)) <= tol
end

# Build a VolumeSource on `datagrid`'s grid using an externally supplied density array
# (e.g. a pointwise product of two orbitals).
function VolumeSource(datagrid, density::AbstractArray{<:Real,3};
                      shift::NTuple{3,<:Real} = (0.0, 0.0, 0.0), tol::Real = 0.0)
    nx, ny, nz = datagrid.nx, datagrid.ny, datagrid.nz
    size(density) == (nx, ny, nz) ||
        throw(ArgumentError("density size $(size(density)) != grid ($nx,$ny,$nz)"))
    T = Float64
    axes = (collect(T((i - 1) / nx) for i in 1:nx),
            collect(T((j - 1) / ny) for j in 1:ny),
            collect(T((k - 1) / nz) for k in 1:nz))
    At, Bt, Ct = true_cell_vectors(datagrid)
    basis = ((At[1], At[2], At[3]), (Bt[1], Bt[2], Bt[3]), (Ct[1], Ct[2], Ct[3]))
    jac = abs(det(hcat(collect(At), collect(Bt), collect(Ct))))
    weights = fill(jac / (nx * ny * nz), nx, ny, nz)
    dens = Array{T,3}(density)
    shift_f = (Float64(shift[1]), Float64(shift[2]), Float64(shift[3]))
    origin = (datagrid.origin[1], datagrid.origin[2], datagrid.origin[3]) .+ shift_f
    return VolumeSource(axes, weights, dens, origin, basis; tol = T(tol))
end
```

- [ ] **Step 4: Create `src/solver/multi_rhs.jl` with `pair_density_source`**

Create `src/solver/multi_rhs.jl`:

```julia
# Multi-RHS per-center batched solve (see docs/.../2026-06-03-per-center-multi-rhs-design.md)

using SparseArrays
using LinearAlgebra
using FMM3D
using Krylov

"""
    pair_density_source(xsf_i, xsf_j; tol=0.0)

Form the pair density rho_ij = phi_i * phi_j as a VolumeSource by reading both orbital
.xsf grids and multiplying pointwise. Requires compatible grids; otherwise resamples
phi_j onto phi_i's grid via trilinear interpolation.
"""
function pair_density_source(xsf_i::AbstractString, xsf_j::AbstractString; tol::Real = 0.0)
    _, dg_i = read_xsf(xsf_i)
    if xsf_i == xsf_j
        return VolumeSource(dg_i, dg_i.values .* dg_i.values; tol = tol)
    end
    _, dg_j = read_xsf(xsf_j)
    if datagrids_compatible(dg_i, dg_j)
        return VolumeSource(dg_i, dg_i.values .* dg_j.values; tol = tol)
    end
    # fallback: resample phi_j onto phi_i's grid
    o, M, Minv, _, _, _ = _datagrid_affine(dg_i)
    prod = similar(dg_i.values)
    nx, ny, nz = dg_i.nx, dg_i.ny, dg_i.nz
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        p = grid_point(dg_i, ix, iy, iz)
        vj = _datagrid_trilinear_value(dg_j, _datagrid_affine(dg_j)[3], p)
        prod[ix, iy, iz] = dg_i.values[ix, iy, iz] * (isnan(vj) ? 0.0 : vj)
    end
    return VolumeSource(dg_i, prod; tol = tol)
end
```

- [ ] **Step 5: Wire into the module**

In `src/BoundaryIntegral.jl`, add after line 93 (`include("solver/dielectric_box3d.jl")`):

```julia
include("solver/multi_rhs.jl")
```

Add exports after line 50:

```julia
export pair_density_source, RHSGroup, assemble_rhs_group, solve_dielectric_box3d_group
```

(`RHSGroup`, `assemble_rhs_group`, `solve_dielectric_box3d_group` are defined in later tasks; exporting now is harmless because the module is loaded as a whole — but to avoid an undefined-export error, only add `pair_density_source` here and add the others in their tasks.)

Replace the line above with just:

```julia
export pair_density_source
```

Register the test in `test/runtests.jl` under the solver group (inside the `if run_full` block is fine, but to keep these fast tests always-on, add outside it near `multi_box3d`):

```julia
    include("solver/multi_rhs.jl")
```

- [ ] **Step 6: Run the test to confirm it passes**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/utils/xsf_reader.jl src/solver/multi_rhs.jl src/BoundaryIntegral.jl test/solver/multi_rhs.jl test/runtests.jl
git commit -m "feat: pair-density rho_ij = phi_i*phi_j from .xsf (Unit 1a)"
```

---

## Task 3: `RHSGroup` assembly (Unit 1b)

**Files:**
- Modify: `src/solver/multi_rhs.jl`
- Modify: `src/BoundaryIntegral.jl` (export `RHSGroup`, `assemble_rhs_group`)
- Modify: `test/solver/multi_rhs.jl`

The `RHSGroup` keeps all densities on the **shared grid** (identical positions/weights across columns) so that the FMM `nd` batching works downstream.

- [ ] **Step 1: Write the failing test**

Append to `test/solver/multi_rhs.jl` inside the `@testset "multi_rhs"`:

```julia
    @testset "assemble_rhs_group" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        # center 1 groups only with itself under cutoff 3.0
        g = assemble_rhs_group(si, 1)
        @test g.center_id == 1
        @test g.neighbor_ids == [1]
        @test size(g.densities, 2) == 1           # K = 1
        @test size(g.positions, 2) == size(g.densities, 1)
        @test all(g.densities[:, 1] .≈ 1.0)        # rho_11 = 1
    end
```

- [ ] **Step 2: Run it to confirm failure**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: FAIL — `assemble_rhs_group` not defined.

- [ ] **Step 3: Implement `RHSGroup` and `assemble_rhs_group`**

Append to `src/solver/multi_rhs.jl`:

```julia
"""
    RHSGroup

One center's batch of pair densities, all on a shared grid.
- `center_id`     : the center orbital id i
- `neighbor_ids`  : the j ids (length K), the columns of `densities`
- `positions`     : 3 x n shared grid points
- `weights`       : n quadrature weights (shared)
- `densities`     : n x K, column k is rho_{i, neighbor_ids[k]}
"""
struct RHSGroup
    center_id::Int
    neighbor_ids::Vector{Int}
    positions::Matrix{Float64}
    weights::Vector{Float64}
    densities::Matrix{Float64}
end

num_pairs(g::RHSGroup) = length(g.neighbor_ids)

function assemble_rhs_group(si::SystemInput, center_id::Int; tol::Real = 0.0)
    haskey(si.groups, center_id) || error("no group for center $center_id")
    js = si.groups[center_id]
    orb_i = si.orbitals[center_id]
    # build each rho_ij on phi_i's grid (positions identical across the group)
    first_vs = pair_density_source(orb_i.xsf_path, si.orbitals[js[1]].xsf_path; tol = tol)
    n = length(first_vs.density)
    K = length(js)
    positions = copy(first_vs.positions)
    weights = copy(first_vs.weights)
    densities = Matrix{Float64}(undef, n, K)
    densities[:, 1] .= first_vs.density
    for k in 2:K
        vs = pair_density_source(orb_i.xsf_path, si.orbitals[js[k]].xsf_path; tol = tol)
        length(vs.density) == n ||
            error("group $center_id: pair $(js[k]) has $(length(vs.density)) pts, expected $n " *
                  "(grids must match across the group for nd-batching)")
        densities[:, k] .= vs.density
    end
    return RHSGroup(center_id, copy(js), positions, weights, densities)
end
```

- [ ] **Step 4: Export and run**

In `src/BoundaryIntegral.jl`, change the multi_rhs export line to:

```julia
export pair_density_source, RHSGroup, assemble_rhs_group
```

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solver/multi_rhs.jl src/BoundaryIntegral.jl test/solver/multi_rhs.jl
git commit -m "feat: RHSGroup assembly on shared grid (Unit 1b)"
```

---

## Task 4: Envelope source + shared group panelization (Unit 2)

**Files:**
- Modify: `src/solver/multi_rhs.jl` (add `envelope_volume_source`, `build_group_interface`)
- Modify: `test/solver/multi_rhs.jl`

The union/envelope refinement reuses the existing `multi_dielectric_box3d_rhs_adaptive(n_quad, l_ec, boxes, epses, vs::VolumeSource, rhs_atol; eps_out, …)` (`src/shape/box3d_multi.jl:437`), driven by an **envelope source** whose density is the per-point root-sum-square of the group's densities. Resolving the envelope resolves every member.

- [ ] **Step 1: Write the failing test**

Append to `test/solver/multi_rhs.jl`:

```julia
    @testset "envelope + group interface" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        env = BoundaryIntegral.envelope_volume_source(g)
        @test length(env.density) == size(g.densities, 1)
        @test all(env.density .≈ 1.0)              # rss of a single all-ones column

        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)
        @test interface isa DielectricInterface
        @test length(interface.panels) >= 6        # at least the 6 box faces
    end
```

- [ ] **Step 2: Run it to confirm failure**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: FAIL — `envelope_volume_source` / `build_group_interface` not defined.

- [ ] **Step 3: Implement**

Append to `src/solver/multi_rhs.jl`:

```julia
"""
    envelope_volume_source(g::RHSGroup)

Per-point root-sum-square of the group's densities, as a VolumeSource on the shared grid.
Drives the union/envelope refinement so the single panelization resolves every member.
"""
function envelope_volume_source(g::RHSGroup)
    n = size(g.densities, 1)
    env = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        acc = 0.0
        for k in 1:size(g.densities, 2)
            acc += g.densities[s, k]^2
        end
        env[s] = sqrt(acc)
    end
    return VolumeSource(copy(g.positions), copy(g.weights), env)
end

"""
    build_group_interface(si, g; n_quad, rhs_atol, l_ec, eps_out=si.eps_out, kw...)

One shared union/envelope-refined DielectricInterface for the whole group.
"""
function build_group_interface(si::SystemInput, g::RHSGroup;
        n_quad::Int, rhs_atol::Float64, l_ec::Float64,
        eps_out::Float64 = si.eps_out, max_depth::Int = 128)
    env = envelope_volume_source(g)
    return multi_dielectric_box3d_rhs_adaptive(
        n_quad, l_ec, si.boxes, si.epses, env, rhs_atol;
        eps_out = eps_out, max_depth = max_depth)
end
```

- [ ] **Step 4: Export and run**

In `src/BoundaryIntegral.jl`, extend the export line:

```julia
export pair_density_source, RHSGroup, assemble_rhs_group, build_group_interface
```

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/solver/multi_rhs.jl src/BoundaryIntegral.jl test/solver/multi_rhs.jl
git commit -m "feat: envelope source + shared group panelization (Unit 2)"
```

---

## Task 5: Batched RHS assembly (Unit 4)

**Files:**
- Modify: `src/solver/dielectric_box3d.jl` (add `rhs_dielectric_box3d_fmm3d_batched`)
- Modify: `src/BoundaryIntegral.jl` (export)
- Modify: `test/solver/multi_rhs.jl`

This must reproduce the existing single-RHS FMM path column-by-column (the convention-pinning regression).

- [ ] **Step 1: Write the failing regression test**

Append to `test/solver/multi_rhs.jl`. It builds a group interface and checks that the batched RHS column equals the existing single-source `rhs_dielectric_box3d_fmm3d` for the same screened density.

```julia
    @testset "batched RHS == single-RHS (K=1 regression)" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)

        F = rhs_dielectric_box3d_fmm3d_batched(interface, si, g, 1e-9)
        @test size(F) == (BoundaryIntegral.num_points(interface), 1)

        # reference: the existing per-source screened FMM RHS for rho_11
        vs = VolumeSource(copy(g.positions), copy(g.weights), g.densities[:, 1])
        f_ref = rhs_dielectric_box3d_fmm3d(interface, vs, 1e-9)   # screened convenience method
        @test maximum(abs.(F[:, 1] .- f_ref)) < 1e-6
    end
```

(Confirm the exact 2-arg `rhs_dielectric_box3d_fmm3d(interface, vs, thresh)` screened convenience method exists at `src/solver/dielectric_box3d.jl:162`; if the screened method takes a different arity, match it. The point is column-for-column agreement.)

- [ ] **Step 2: Run it to confirm failure**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: FAIL — `rhs_dielectric_box3d_fmm3d_batched` not defined.

- [ ] **Step 3: Implement the batched RHS builder**

In `src/solver/dielectric_box3d.jl`, after the multi-box `rhs_dielectric_box3d_fmm3d` methods (end of file), add:

```julia
# Batched RHS for a whole center group on a shared grid.
# Returns an N x K matrix F whose column k is f_{i, neighbor_ids[k]} = -∂n u_inc[rho_ij].
function rhs_dielectric_box3d_fmm3d_batched(
    interface::DielectricInterface{P, Float64},
    si::SystemInput,
    group::RHSGroup,
    thresh::Float64,
) where {P <: AbstractPanel}
    n = size(group.positions, 2)
    K = num_pairs(group)
    n_points = num_points(interface)

    # per-point screening factor 1/eps_local (identical across columns -> nd-batchable)
    inv_eps = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        pos = (group.positions[1, s], group.positions[2, s], group.positions[3, s])
        eps_local = si.eps_out
        for b in eachindex(si.boxes)
            box = si.boxes[b]
            lo = (box.center[1] - box.Lx/2, box.center[2] - box.Ly/2, box.center[3] - box.Lz/2)
            hi = (box.center[1] + box.Lx/2, box.center[2] + box.Ly/2, box.center[3] + box.Lz/2)
            if lo[1] <= pos[1] <= hi[1] && lo[2] <= pos[2] <= hi[2] && lo[3] <= pos[3] <= hi[3]
                eps_local = si.epses[b]; break
            end
        end
        inv_eps[s] = 1.0 / eps_local
    end

    # charges (nd=K, n) = weight * (1/eps) * density, per column
    charges = Matrix{Float64}(undef, K, n)
    @inbounds for k in 1:K, s in 1:n
        charges[k, s] = group.weights[s] * inv_eps[s] * group.densities[s, k]
    end

    targets = zeros(Float64, 3, n_points)
    normals = zeros(Float64, 3, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        targets[1, i] = point.panel_point.point[1]
        targets[2, i] = point.panel_point.point[2]
        targets[3, i] = point.panel_point.point[3]
        normals[1, i] = point.panel_point.normal[1]
        normals[2, i] = point.panel_point.normal[2]
        normals[3, i] = point.panel_point.normal[3]
    end

    vals = lfmm3d(thresh, group.positions, charges = charges, targets = targets, pgt = 2)
    grad = vals.gradtarg   # (K, 3, n_points)
    F = Matrix{Float64}(undef, n_points, K)
    @inbounds for k in 1:K, i in 1:n_points
        F[i, k] = (normals[1, i] * grad[k, 1, i] +
                   normals[2, i] * grad[k, 2, i] +
                   normals[3, i] * grad[k, 3, i]) / (4π)
    end
    return F
end
```

Sign/scaling note: this mirrors `rhs_dielectric_box3d_fmm3d` (`/(4π·eps_src)` with the screened density folding `1/eps` and `eps_src=1`). The Step-1 regression test is authoritative — if it disagrees in sign, flip `F` to match `f_ref`.

- [ ] **Step 4: Export and run**

In `src/BoundaryIntegral.jl`, add to the `rhs_dielectric_box3d*` export line:

```julia
export rhs_dielectric_box3d, rhs_dielectric_box3d_fmm3d, rhs_dielectric_box3d_hybrid, rhs_dielectric_box3d_fmm3d_batched
```

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: PASS (the batched column matches the single-source reference to < 1e-6).

- [ ] **Step 5: Commit**

```bash
git add src/solver/dielectric_box3d.jl src/BoundaryIntegral.jl test/solver/multi_rhs.jl
git commit -m "feat: nd-batched RHS assembly with K=1 regression (Unit 4)"
```

---

## Task 6: Vectorized matrix-capable operator (Unit 5) ⭐

**Files:**
- Modify: `src/solver/multi_rhs.jl` (add `BatchedDielectricOperator`, constructor, `mul!`, `size`, `*`)
- Modify: `test/solver/multi_rhs.jl`

The operator carries its own FMM geometry + sparse corrections + diagonal so it can do **one batched FMM per block matvec**. It must agree with the existing per-vector LinearMap operator column-by-column.

- [ ] **Step 1: Write the failing test**

Append to `test/solver/multi_rhs.jl`:

```julia
    @testset "batched matvec == columnwise (Unit 5)" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)

        Np = BoundaryIntegral.num_points(interface)
        op  = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-9, 1e-9, 8)
        ref = lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-9, 1e-9, 8)

        # vector path matches the reference LinearMap
        x = collect(range(0.1, 1.0; length = Np))
        @test maximum(abs.(op * x .- ref * x)) < 1e-9

        # matrix path equals applying the reference to each column
        X = hcat(x, reverse(x), fill(0.5, Np))
        Y = op * X
        @test size(Y) == size(X)
        for c in 1:size(X, 2)
            @test maximum(abs.(Y[:, c] .- ref * X[:, c])) < 1e-8
        end
    end
```

- [ ] **Step 2: Run it to confirm failure**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: FAIL — `batched_lhs_dielectric_box3d_fmm3d_corrected` not defined.

- [ ] **Step 3: Implement the operator**

Append to `src/solver/multi_rhs.jl`:

```julia
"""
    BatchedDielectricOperator

Matrix-capable adjoint-double-layer + diagonal operator for the dielectric BIE. A single
batched FMM (`nd = K`) serves an N x K block; the sparse near correction and the diagonal
contrast term are applied as batched mat-mat / broadcasts.
"""
struct BatchedDielectricOperator
    sources::Matrix{Float64}        # 3 x n
    weights::Vector{Float64}        # n
    norms::Matrix{Float64}          # 3 x n
    thresh::Float64
    corrections::SparseMatrixCSC{Float64,Int}
    diag::Vector{Float64}           # n
    n::Int
end

Base.size(op::BatchedDielectricOperator) = (op.n, op.n)
Base.size(op::BatchedDielectricOperator, d::Integer) = d <= 2 ? op.n : 1
Base.eltype(::BatchedDielectricOperator) = Float64

function batched_lhs_dielectric_box3d_fmm3d_corrected(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64, up_tol::Float64, max_order::Int;
    range_factor::Float64 = 5.0, correct_edges::Bool = false,
    adaptive_atol::Float64 = up_tol, adaptive_rtol::Float64 = sqrt(eps(Float64)),
    adaptive_n_GL::Int = 0, adaptive_max_depth::Int = 20,
) where {P <: AbstractPanel}
    n = num_points(interface)
    sources = zeros(Float64, 3, n)
    weights = zeros(Float64, n)
    norms = zeros(Float64, 3, n)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
        sources[3, i] = point.panel_point.point[3]
        norms[1, i] = point.panel_point.normal[1]
        norms[2, i] = point.panel_point.normal[2]
        norms[3, i] = point.panel_point.normal[3]
    end

    adaptive_cfg = AdaptiveConfig(adaptive_atol, adaptive_rtol, adaptive_n_GL, adaptive_max_depth)
    (; upsample, adaptive) = build_neighbor_list(interface, max_order, up_tol;
        range_factor = range_factor, correct_edges = correct_edges, adaptive_cfg = adaptive_cfg)
    corrections = laplace3d_DT_corrections(interface, upsample, adaptive)

    diag = Vector{Float64}(undef, n)
    offset = 0
    for i in 1:length(interface.panels)
        eps_in = interface.eps_in[i]; eps_out = interface.eps_out[i]
        np = num_points(interface.panels[i])
        t = 0.5 * (eps_out + eps_in) / (eps_out - eps_in)
        for j in 1:np
            diag[offset + j] = t
        end
        offset += np
    end

    return BatchedDielectricOperator(sources, weights, norms, fmm_tol,
        SparseMatrixCSC{Float64,Int}(corrections), diag, n)
end

# matrix matvec: one batched FMM for all K columns
function LinearAlgebra.mul!(Y::AbstractMatrix, op::BatchedDielectricOperator, X::AbstractMatrix)
    n = op.n; K = size(X, 2)
    size(X, 1) == n || throw(DimensionMismatch())
    charges = Matrix{Float64}(undef, K, n)
    @inbounds for k in 1:K, i in 1:n
        charges[k, i] = op.weights[i] * X[i, k]
    end
    vals = lfmm3d(op.thresh, op.sources, charges = charges, pg = 2)
    grad = vals.grad                      # (K, 3, n)
    C = op.corrections * X                # n x K (sparse mat-mat)
    @inbounds for k in 1:K, i in 1:n
        gn = op.norms[1, i] * grad[k, 1, i] +
             op.norms[2, i] * grad[k, 2, i] +
             op.norms[3, i] * grad[k, 3, i]
        Y[i, k] = -gn / (4π) + C[i, k] + op.diag[i] * X[i, k]
    end
    return Y
end

function LinearAlgebra.mul!(y::AbstractVector, op::BatchedDielectricOperator, x::AbstractVector)
    Y = reshape(y, op.n, 1)
    mul!(Y, op, reshape(x, op.n, 1))
    return y
end

Base.:*(op::BatchedDielectricOperator, X::AbstractMatrix) =
    mul!(Matrix{Float64}(undef, op.n, size(X, 2)), op, X)
Base.:*(op::BatchedDielectricOperator, x::AbstractVector) =
    mul!(Vector{Float64}(undef, op.n), op, x)
```

Note the `-gn/(4π)` matches `laplace3d_DT_fmm3d` (`src/kernel/laplace3d.jl:106`, `return -gradn ./ 4π`). The columnwise test pins this.

- [ ] **Step 4: Export and run**

In `src/BoundaryIntegral.jl`, extend the lhs export line (after line 49) to include:

```julia
export batched_lhs_dielectric_box3d_fmm3d_corrected, BatchedDielectricOperator
```

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: PASS (vector and all matrix columns agree with the reference LinearMap).

- [ ] **Step 5: Commit**

```bash
git add src/solver/multi_rhs.jl src/BoundaryIntegral.jl test/solver/multi_rhs.jl
git commit -m "feat: vectorized nd-batched matrix-capable BIE operator (Unit 5)"
```

---

## Task 7: Block GMRES group solve driver (Unit 6) + integration

**Files:**
- Modify: `src/solver/multi_rhs.jl` (add `solve_dielectric_box3d_group`)
- Modify: `src/BoundaryIntegral.jl` (export)
- Modify: `test/solver/multi_rhs.jl`

- [ ] **Step 1: Write the failing test (block vs looped + end-to-end)**

Append to `test/solver/multi_rhs.jl`:

```julia
    @testset "block solve == looped single solves (Unit 6)" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_small.bie"))
        g = assemble_rhs_group(si, 1)
        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)
        op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-9, 1e-9, 8)
        F = rhs_dielectric_box3d_fmm3d_batched(interface, si, g, 1e-9)

        # block solve
        Σ, stats = BoundaryIntegral._block_gmres_solve(op, F; rtol = 1e-10, itmax = 200)
        @test size(Σ) == size(F)

        # reference: loop single gmres per column
        using Krylov
        for c in 1:size(F, 2)
            xc, _ = Krylov.gmres(op, F[:, c]; rtol = 1e-10, itmax = 200)
            @test maximum(abs.(Σ[:, c] .- xc)) < 1e-6
        end
    end

    @testset "end-to-end .bie -> Sigma" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        out = solve_dielectric_box3d_group(
            joinpath(fixdir, "system_small.bie"), 1;
            n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25,
            fmm_tol = 1e-9, up_tol = 1e-9, max_order = 8, rtol = 1e-10)
        @test size(out.sigma, 2) == 1
        @test size(out.sigma, 1) == BoundaryIntegral.num_points(out.interface)
        @test all(isfinite, out.sigma)
    end
```

- [ ] **Step 2: Run it to confirm failure**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: FAIL — `_block_gmres_solve` / `solve_dielectric_box3d_group` not defined.

- [ ] **Step 3: Implement the driver**

Append to `src/solver/multi_rhs.jl`:

```julia
# thin wrapper so tests can call block_gmres directly on the operator
function _block_gmres_solve(op::BatchedDielectricOperator, F::AbstractMatrix;
        rtol::Float64 = 1e-10, atol::Float64 = 0.0, itmax::Int = 500)
    Σ, stats = Krylov.block_gmres(op, Matrix{Float64}(F); rtol = rtol, atol = atol, itmax = itmax)
    return Σ, stats
end

"""
    solve_dielectric_box3d_group(bie_path, center_id; kw...) -> (; sigma, interface, group, stats)

Step 0 + Steps 1-6: parse the .bie file, assemble the center group, build the shared
union/envelope-refined interface, assemble the batched RHS, and block-GMRES solve for the
layer densities Σ (N x K). Stops at Σ (evaluation/contraction are out of scope).
"""
function solve_dielectric_box3d_group(bie_path::AbstractString, center_id::Int;
        n_quad::Int, rhs_atol::Float64, l_ec::Float64,
        fmm_tol::Float64 = 1e-9, up_tol::Float64 = 1e-9, max_order::Int = 8,
        rtol::Float64 = 1e-10, itmax::Int = 500, tol::Real = 0.0,
        max_depth::Int = 128)
    si = read_system_input(bie_path)
    group = assemble_rhs_group(si, center_id; tol = tol)
    isempty(group.neighbor_ids) &&
        return (; sigma = zeros(0, 0), interface = nothing, group = group, stats = nothing)
    interface = build_group_interface(si, group;
        n_quad = n_quad, rhs_atol = rhs_atol, l_ec = l_ec, max_depth = max_depth)
    op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, fmm_tol, up_tol, max_order)
    F = rhs_dielectric_box3d_fmm3d_batched(interface, si, group, fmm_tol)
    Σ, stats = _block_gmres_solve(op, F; rtol = rtol, itmax = itmax)
    return (; sigma = Σ, interface = interface, group = group, stats = stats)
end
```

- [ ] **Step 4: Export and run**

In `src/BoundaryIntegral.jl`, extend the multi_rhs export line:

```julia
export pair_density_source, RHSGroup, assemble_rhs_group, build_group_interface,
    batched_lhs_dielectric_box3d_fmm3d_corrected, BatchedDielectricOperator,
    solve_dielectric_box3d_group
```

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: PASS (block matches looped solves; end-to-end produces a finite N×1 Σ).

- [ ] **Step 5: Run the full default test suite to confirm no regressions**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS, including the new `utils/system_input.jl` and `solver/multi_rhs.jl`.

- [ ] **Step 6: Commit**

```bash
git add src/solver/multi_rhs.jl src/BoundaryIntegral.jl test/solver/multi_rhs.jl
git commit -m "feat: block-GMRES per-center group solve, end-to-end (Unit 6)"
```

---

## Task 8: Multi-pair fixture + true K>1 batch test

**Files:**
- Create: `test/fixtures/system_pair.bie`, `test/fixtures/orb_c.xsf`
- Modify: `test/solver/multi_rhs.jl`

The earlier fixtures only exercise K=1. This adds a K=2 group to verify true batching: two overlapping orbitals on the same grid.

- [ ] **Step 1: Create K=2 fixtures**

`test/fixtures/orb_c.xsf`: same grid as `orb_a.xsf`, atom at `X 0.5 0.0 0.0`, density values a non-constant pattern, e.g. the 8 grid values `0.2 0.4 0.6 0.8 0.3 0.5 0.7 0.9` (any distinct finite values). `test/fixtures/system_pair.bie`:

```
UNITS bohr
BEGIN_DIELECTRICS
EPS_OUT 1.0
  0.0 0.0 0.0    4.0 4.0 4.0    11.7
END_DIELECTRICS
BEGIN_ORBITALS
  1   orb_a.xsf
  3   orb_c.xsf
END_ORBITALS
BEGIN_GROUPING
CUTOFF 10.0
END_GROUPING
```

(Cutoff 10.0 → center 1 groups with both 1 and 3, so K=2.)

- [ ] **Step 2: Write the K=2 test**

Append to `test/solver/multi_rhs.jl`:

```julia
    @testset "K=2 batched solve matches per-column" begin
        fixdir = joinpath(@__DIR__, "..", "fixtures")
        si = read_system_input(joinpath(fixdir, "system_pair.bie"))
        g = assemble_rhs_group(si, 1)
        @test num_pairs(g) == 2

        interface = build_group_interface(si, g; n_quad = 6, rhs_atol = 1e-3, l_ec = 0.25)
        op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-9, 1e-9, 8)
        F = rhs_dielectric_box3d_fmm3d_batched(interface, si, g, 1e-9)
        @test size(F, 2) == 2

        Σ, _ = BoundaryIntegral._block_gmres_solve(op, F; rtol = 1e-10, itmax = 300)
        for c in 1:2
            xc, _ = Krylov.gmres(op, F[:, c]; rtol = 1e-10, itmax = 300)
            @test maximum(abs.(Σ[:, c] .- xc)) < 1e-6
        end
    end
```

- [ ] **Step 3: Run it to confirm pass**

Run: `julia --project -e 'using BoundaryIntegral, Test; include("test/solver/multi_rhs.jl")'`
Expected: PASS — K=2 block solve matches the two independent single solves.

- [ ] **Step 4: Commit**

```bash
git add test/fixtures/system_pair.bie test/fixtures/orb_c.xsf test/solver/multi_rhs.jl
git commit -m "test: K=2 batched group solve matches per-column gmres"
```

---

## Self-Review

**Spec coverage:**
- Step 0 input format → Task 1 (parser, fixtures, malformed-input error). ✓
- Unit 1 (pair density `phi_i*phi_j`, grid validate + resample fallback) → Task 2; `RHSGroup` shared-grid assembly → Task 3. ✓
- Unit 2 (union/envelope refinement, reuse `multi_dielectric_box3d_rhs_adaptive`) → Task 4. ✓
- Unit 3 (build A once) → reused inside Task 6 operator construction. ✓
- Unit 4 (batched RHS, nd FMM, K=1 regression) → Task 5. ✓
- Unit 5 (matrix-capable vectorized matvec, columnwise agreement) → Task 6. ✓
- Unit 6 (block GMRES, block-vs-looped, end-to-end) → Task 7; true K>1 batch → Task 8. ✓
- Tests 0–5 in the spec's testing strategy all map to tasks. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test shows full assertions. The two convention notes (RHS sign in Task 5, DT sign in Task 6) are pinned by regression tests, not left vague. ✓

**Type consistency:** `SystemInput` fields (`unit_scale, boxes, epses, eps_out, orbitals, groups`) used identically in Tasks 1/3/4/5/7. `RHSGroup` fields (`center_id, neighbor_ids, positions, weights, densities`) consistent across Tasks 3–8. `BatchedDielectricOperator` constructor name `batched_lhs_dielectric_box3d_fmm3d_corrected` and `mul!`/`*` used consistently in Tasks 6–8. `pair_density_source`, `assemble_rhs_group`, `build_group_interface`, `envelope_volume_source`, `rhs_dielectric_box3d_fmm3d_batched`, `_block_gmres_solve`, `solve_dielectric_box3d_group` names consistent throughout. ✓

**Known risk to watch during execution:** the `.xsf` fixture datagrid block format must parse under `read_xsf` (it keys on `BEGIN_DATAGRID_3D` prefix and reads 6 header lines then values). If `read_xsf` rejects the hand-written fixture, adjust the fixture to match the exact block layout `read_xsf` expects (see `src/utils/xsf_reader.jl:66-98`) before writing implementation code.
