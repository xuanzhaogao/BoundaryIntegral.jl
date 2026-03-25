# BoundaryIntegral.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://arrogantgao.github.io/BoundaryIntegral.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://arrogantgao.github.io/BoundaryIntegral.jl/dev/)
[![Build Status](https://github.com/ArrogantGao/BoundaryIntegral.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/BoundaryIntegral.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ArrogantGao/BoundaryIntegral.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/BoundaryIntegral.jl)

BoundaryIntegral.jl provides boundary integral operators and solvers for Laplace problems in 2D and 3D, with support for dielectric interfaces, panel-based quadrature, and fast multipole acceleration.

## Features

- Laplace single- and double-layer operators in 2D and 3D.
- Dielectric interface builders for box geometries.
- Direct and FMM-accelerated linear operators for iterative solves.
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
