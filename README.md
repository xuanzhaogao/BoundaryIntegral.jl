# BoundaryIntegral.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xuanzhaogao.github.io/BoundaryIntegral.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xuanzhaogao.github.io/BoundaryIntegral.jl/dev/)
[![Build Status](https://github.com/xuanzhaogao/BoundaryIntegral.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xuanzhaogao/BoundaryIntegral.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/xuanzhaogao/BoundaryIntegral.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xuanzhaogao/BoundaryIntegral.jl)

BoundaryIntegral.jl solves Laplace problems with dielectric interfaces in 2D and 3D using
boundary integral equations. It provides single- and double-layer operators, panel-based
quadrature with adaptive refinement, FMM acceleration with near-field correction, volume
charge sources, many-right-hand-side (block) solves, and a batched four-index integral
pipeline that can run distributed across a cluster.

## Features

- Laplace single- and double-layer operators in 2D and 3D.
- Dielectric interface builders for box geometries, with adaptive panel refinement.
- Direct and FMM-accelerated linear operators (with near-field correction) for iterative solves.
- Volume charge sources, plus a reusable precomputed spectral field (`PrecomputedVolumeField`)
  for fast repeated potential/gradient evaluation and right-hand-side assembly.
- Many-right-hand-side (block) solves and four-index Coulomb integrals.
- A batched lattice campaign pipeline for large four-index runs, with optional
  distributed/Slurm execution.
- Linear algebra helpers (`solve_lu`, `solve_gmres`).
- Optional Makie visualization extension.

The public API is exported, so `using BoundaryIntegral` brings the functions below into
scope directly.

## Installation

```julia
using Pkg
Pkg.add("BoundaryIntegral")
```

For local development:

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Quickstart (2D)

```julia
using BoundaryIntegral

# single_dielectric_box2d(Lx, Ly, n_quad, l_panel, l_corner, eps_in, eps_out[, T])
interface = single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64)
lhs = lhs_dielectric_box2d(interface)
rhs = rhs_dielectric_box2d(interface, PointSource((0.1, 0.1), 1.0), 5.0)

sigma = solve_lu(lhs, rhs)
```

## Quickstart (3D)

```julia
using BoundaryIntegral

# single_dielectric_box3d(Lx, Ly, Lz, n_quad, l_ec, eps_in, eps_out[, T])
interface = single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, 4.0, 1.0, Float64)
lhs = lhs_dielectric_box3d(interface)
rhs = rhs_dielectric_box3d(interface, PointSource((0.1, 0.1, 0.1), 1.0), 4.0)

sigma = solve_lu(lhs, rhs)
```

## FMM-accelerated solves

For larger problems, replace the dense operator with an FMM-accelerated `LinearMap` and
solve with GMRES. In 3D, `lhs_dielectric_box3d_fmm3d_corrected` adds the near-field
correction needed for accuracy near the surface.

```julia
using BoundaryIntegral

interface = single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64)
lhs_fmm = lhs_dielectric_box2d_fmm2d(interface, 1e-12)
rhs = rhs_dielectric_box2d(interface, PointSource((0.1, 0.1), 1.0), 5.0)

# solve_gmres(A, b, atol, rtol)
sigma = solve_gmres(lhs_fmm, rhs, 1e-12, 1e-12)
```

## Volume sources and precomputed fields

A `VolumeSource` is a volume charge density sampled on a quadrature grid: positions
(`3 × N`), quadrature weights, and density values. Its free-space Laplace potential can
drive the right-hand side of a dielectric problem on an interface that is adaptively
refined to resolve that source:

```julia
using BoundaryIntegral

# positions: 3×N matrix, weights/density: length-N vectors
vs = VolumeSource(positions, weights, density)

# single_dielectric_box3d_rhs_adaptive(Lx, Ly, Lz, n_quad, source,
#                                       eps_src, l_ec, rhs_atol, eps_in, eps_out[, T])
interface = single_dielectric_box3d_rhs_adaptive(
    Lx, Ly, Lz, n_quad, vs, eps_src, l_ec, rhs_atol, eps_in, eps_out, Float64)

rhs = rhs_dielectric_box3d_hybrid(interface, vs, eps_src, 1e-6)
```

When the same source is reused for many target batches or assemblies, build the field
once and reuse it. `PrecomputedVolumeField` stores a truncated spectral representation;
in-box targets are evaluated with a type-2 NUFFT and out-of-box targets with the FMM:

```julia
field = PrecomputedVolumeField(vs; tol = 1e-6)

phi  = volume_field_potential(field, targets)   # targets: 3×n  ->  length-n potential
grad = volume_field_gradient(field, targets)    #              ->  3×n gradient
rhs  = rhs_dielectric_box3d_field(interface, field, eps_src)
```

## Many right-hand sides and four-index integrals

For several sources sharing one interface, assemble and block-solve all right-hand sides
at once, then contract into the four-index Coulomb matrix `V[a, b]`:

```julia
using BoundaryIntegral

# vss::Vector{VolumeSource} on a shared interface that resolves all of them
# (e.g. built with multi_dielectric_box3d_rhs_adaptive)
sigma, stats = solve_dielectric_box3d_block(interface, vss)
V = four_index_matrix(interface, vss, sigma; lhs_tol = 1e-6, volume_tol = 1e-6)
```

## Lattice campaign (batched, optionally distributed)

For large four-index runs over many orbital centers on a lattice, the package provides a
file-backed pipeline driven from a TOML campaign description. It groups pair densities
into batches that share one boundary operator, solves each batch, then evaluates and
contracts every pair against a shared target set.

### Pipeline

```
prepare  →  solve_batch  →  consolidate  →  eval_batch  →  assemble_v
```

- **prepare** — enumerate centers, neighbor pairs, and batches; write the manifest.
- **solve_batch** — per batch: assemble the shared interface, block-GMRES, store σ + screened ρ.
- **consolidate** — build the shared evaluation target set and the contraction store.
- **eval_batch** — per batch: evaluate Φ at the shared targets and contract all pairs into V columns.
- **assemble_v** — gather all V columns into the dense matrix; write `V_full.tsv` + `report.txt`.

Every phase writes its outputs atomically (temp file + rename), so a run is crash-safe and
restartable: re-running a phase skips already-completed batches, and a killed job is
recovered simply by rerunning the phase. `pending_batches(c, :solve | :eval)` reports
what is left to do.

### Campaign TOML

```toml
name = "demo"                          # campaign name
root = "/path/to/output"               # output dir: manifest.tsv, batches/, V/, logs/
templates = ["orb1.xsf", "orb2.xsf"]   # type index -> .xsf path (relative to this file)

[[orbital]]                            # one entry per orbital; id = 1-based order
type = 1                               # index into `templates`
x = 0.0                                # Cartesian center in the templates' frame
y = 0.0
z = 7.5

[pairing]
neighbor_cutoff = 2.6                  # pair orbitals within this distance (default: Inf)
# pairs = [[1, 2], [1, 3]]             # OR give explicit pair overrides instead

[dielectrics]
eps_out = 1.0
boxes = [[0.0, 0.0, 7.5, 90.0, 90.0, 3.35, 3.5]]   # rows of [cx cy cz Lx Ly Lz eps]

[solve]
n_quad = 6
edge_refine_level = 2                  # or set `l_ec` directly
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

### Running serially

```julia
using BoundaryIntegral

c = load_campaign("campaign.toml")
prepare(c)
for id in pending_batches(c, :solve); solve_batch(c, id); end
consolidate(c)
for id in pending_batches(c, :eval); eval_batch(c, id); end
assemble_v(c)
```

The whole pipeline can also be run in memory, without writing files, via
`four_index_integrals("campaign.toml")` (returns `(; pair_ids, V)`).

### Running distributed

The `:solve` and `:eval` phases parallelize over batches. `run_phase` is provided by a
package extension; load its weak dependencies (`Distributed` and `SlurmClusterManager`) to
enable it. Spawning policy: inside a Slurm allocation with more than one task it uses
`SlurmManager()` (one worker per task); otherwise it spawns `workers` local processes; with
`workers = 0` and no allocation it runs inline.

```julia
using BoundaryIntegral, Distributed, SlurmClusterManager

c = load_campaign("campaign.toml")
prepare(c)
run_phase(c, :solve; workers = 4)   # local workers (or SlurmManager inside an allocation)
consolidate(c)
run_phase(c, :eval; workers = 4)
assemble_v(c)
```

A campaign is typically driven by a small script over this API (the package itself ships
no CLI). On Slurm, run one task per node and give each task the whole node via threads:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=logs/solve_%j.out
set -euo pipefail

# Pin both thread pools — unpinned threads silently corrupt timings.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_GLUE_THREADS=8

julia --project -t "$JULIA_GLUE_THREADS" driver.jl campaign.toml solve
```

where `driver.jl` parses `<campaign.toml> <phase>` and calls the phase functions above.
Notes for cluster runs:

- **Precompile on the head process first** (`using Pkg; Pkg.precompile()` before loading
  `Distributed`/`SlurmClusterManager`) so workers don't race to precompile over a shared
  filesystem.
- **One worker per node** (`--ntasks-per-node=1`, `--cpus-per-task=<cores>`); the FMM
  saturates many cores, so node-sized tasks are the right grain.
- **Crash/walltime recovery:** just resubmit the same phase — completed batches are skipped.

## Visualization

Visualization helpers live under `viz_2d` and `viz_3d`. To use them, install Makie and a
backend, then load the backend before `BoundaryIntegral`:

```julia
using Pkg
Pkg.add(["Makie", "CairoMakie"])
```

```julia
using CairoMakie
using BoundaryIntegral
```

## Development

- Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
  (set `BI_RUN_FULL_TESTS=1` for the full suite, including 3D near-correction and solver tests).
- Build docs: `julia --project=docs docs/make.jl`

## License

MIT. See `LICENSE`.
