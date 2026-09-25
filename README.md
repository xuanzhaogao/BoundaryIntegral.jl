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

## Installation

Requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add("BoundaryIntegral")
```

## Quickstart

```julia
using BoundaryIntegral

# single_dielectric_box3d(Lx, Ly, Lz, n_quad, l_ec, eps_in, eps_out[, T])
interface = single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, 4.0, 1.0, Float64)
lhs = lhs_dielectric_box3d(interface)
rhs = rhs_dielectric_box3d(interface, PointSource((0.1, 0.1, 0.1), 1.0), 4.0)

sigma = solve_lu(lhs, rhs)
```

## Documentation

The [documentation](https://xuanzhaogao.github.io/BoundaryIntegral.jl/dev/) covers:

- [Getting started](https://xuanzhaogao.github.io/BoundaryIntegral.jl/dev/) — 2D and 3D quickstarts.
- [User guide](https://xuanzhaogao.github.io/BoundaryIntegral.jl/dev/guide/) — FMM-accelerated solves,
  volume sources and precomputed fields, many right-hand sides, four-index integrals, visualization.
- [Lattice campaigns](https://xuanzhaogao.github.io/BoundaryIntegral.jl/dev/campaign/) — batched,
  restartable four-index runs, with optional distributed/Slurm execution.
- [API reference](https://xuanzhaogao.github.io/BoundaryIntegral.jl/dev/api/)
- [Development](https://xuanzhaogao.github.io/BoundaryIntegral.jl/dev/development/) — tests and docs builds.

## License

MIT. See `LICENSE`.
