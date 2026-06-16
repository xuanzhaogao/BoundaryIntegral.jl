# BoundaryIntegral.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xuanzhaogao.github.io/BoundaryIntegral.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xuanzhaogao.github.io/BoundaryIntegral.jl/dev/)
[![Build Status](https://github.com/xuanzhaogao/BoundaryIntegral.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xuanzhaogao/BoundaryIntegral.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/xuanzhaogao/BoundaryIntegral.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xuanzhaogao/BoundaryIntegral.jl)

BoundaryIntegral.jl provides boundary integral operators and solvers for Laplace problems in 2D and 3D, with support for dielectric interfaces, panel-based quadrature, fast multipole acceleration, volume charge sources, many-right-hand-side solves, and a batched four-index integral pipeline that can run distributed across a cluster.

## Features

- Laplace single- and double-layer operators in 2D and 3D.
- Dielectric interface builders for box geometries, with adaptive panel refinement.
- Direct and FMM-accelerated linear operators (with near-field correction) for iterative solves.
- Volume charge sources, plus a reusable precomputed spectral field (`PrecomputedVolumeField`) for fast repeated potential/gradient evaluation and RHS assembly.
- Many-right-hand-side (block) solves and four-index Coulomb integrals.
- A batched lattice campaign pipeline for large four-index runs, with optional distributed/Slurm execution.
- Simple linear algebra helpers (LU and GMRES wrappers).
- Optional Makie visualization extension.

## Installation

In the Julia REPL:

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
import BoundaryIntegral as BI

interface = BI.single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64)
lhs = BI.Lhs_dielectric_box2d(interface)
rhs = BI.Rhs_dielectric_box2d(interface, BI.PointSource((0.1, 0.1), 1.0), 5.0)

sigma = BI.solve_lu(lhs, rhs)
```

## Quickstart (3D)

```julia
using BoundaryIntegral
import BoundaryIntegral as BI

interface = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.4, 0.2, 4.0, 1.0, Float64)
lhs = BI.Lhs_dielectric_box3d(interface)
rhs = BI.Rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), 4.0)

sigma = BI.solve_lu(lhs, rhs)
```

## FMM-Accelerated Solves

```julia
using BoundaryIntegral
import BoundaryIntegral as BI

interface = BI.single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64)
lhs_fmm = BI.Lhs_dielectric_box2d_fmm2d(interface, 1e-12)
rhs = BI.Rhs_dielectric_box2d(interface, BI.PointSource((0.1, 0.1), 1.0), 5.0)

sigma = BI.solve_gmres(lhs_fmm, rhs, 1e-12, 1e-12)
```

## Volume Sources and Precomputed Fields

A `VolumeSource` represents a volume charge density sampled on a quadrature grid
(positions `3 × N`, quadrature weights, and density values). Its free-space Laplace
potential can drive the right-hand side of a dielectric problem on an interface that
is adaptively refined to resolve that source:

```julia
using BoundaryIntegral
import BoundaryIntegral as BI

# volume charge: positions (3×N), quadrature weights, density values
vs = VolumeSource(positions, weights, density)

interface = BI.single_dielectric_box3d_rhs_adaptive(
    Lx, Ly, Lz, n_quad, vs, eps_src, l_ec, rhs_atol, eps_in, eps_out, Float64)

rhs = rhs_dielectric_box3d_hybrid(interface, vs, eps_src, 1e-6)
```

When the same source is reused for many target batches or assemblies, build the field
once and reuse it. `PrecomputedVolumeField` stores a truncated spectral representation;
in-box targets are evaluated with a type-2 NUFFT and out-of-box targets with the FMM:

```julia
field = PrecomputedVolumeField(vs; tol = 1e-6)

phi  = volume_field_potential(field, targets)   # targets: 3×n
grad = volume_field_gradient(field, targets)
rhs  = rhs_dielectric_box3d_field(interface, field, eps_src)
```

## Many Right-Hand Sides and Four-Index Integrals

For several sources sharing one interface, assemble and block-solve all right-hand
sides at once, then contract into the four-index Coulomb matrix `V[a,b]`:

```julia
# vss::Vector{VolumeSource}, on a shared interface that resolves all of them
sigma, stats = solve_dielectric_box3d_block(interface, vss)
V = four_index_matrix(interface, vss, sigma; lhs_tol = 1e-6, volume_tol = 1e-6)
```

## Lattice Campaign (batched, optionally distributed)

For large four-index runs over many orbital centers on a lattice, the package provides
a file-backed pipeline driven from a TOML campaign description. The phases

`prepare` → `solve_batch` → `consolidate` → `eval_batch` → `assemble_v`

write their outputs atomically, so a run is crash-safe and restartable: re-running a
phase skips already-completed batches. A driver is just a small script over the
exported API:

```julia
using BoundaryIntegral
c = load_campaign("campaign.toml")
prepare(c)
for id in pending_batches(c, :solve); solve_batch(c, id); end
consolidate(c)
for id in pending_batches(c, :eval); eval_batch(c, id); end
assemble_v(c)
```

The same computation can be run entirely in memory with
`four_index_integrals("campaign.toml")`.

To run the `:solve`/`:eval` phases across worker processes — locally or inside a Slurm
allocation — load the distributed extension's weak dependencies and call `run_phase`:

```julia
using BoundaryIntegral, Distributed, SlurmClusterManager
run_phase(c, :solve; workers = 4)   # SlurmManager inside an allocation, else local workers
```

## Visualization

Visualization helpers live under `viz_2d` and `viz_3d`. To use them, install Makie and a backend, then load the backend before `BoundaryIntegral`:

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
- Build docs: `julia --project=docs docs/make.jl`

## License

MIT. See `LICENSE`.
