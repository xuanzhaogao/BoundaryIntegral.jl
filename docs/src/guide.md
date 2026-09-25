```@meta
CurrentModule = BoundaryIntegral
```

# User guide

This page covers the solver features beyond the [quickstart](index.md): FMM acceleration,
volume sources, many-right-hand-side solves, and visualization. For large distributed
four-index runs, see [Lattice campaigns](campaign.md).

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

# A Gaussian charge sampled on a uniform midpoint grid: positions (3×N), weights, density.
n, h = 24, 1.2 / 24
xs = range(-0.6 + h / 2, 0.6 - h / 2; length = n)
positions = reduce(hcat, [[x, y, z] for x in xs for y in xs for z in xs])
density = [exp(-sum(abs2, p) / (2 * 0.08^2)) for p in eachcol(positions)]
vs = VolumeSource(positions, fill(h^3, n^3), density)

# single_dielectric_box3d_rhs_adaptive(Lx, Ly, Lz, n_quad, source,
#                                       eps_src, l_ec, rhs_atol, eps_in, eps_out[, T])
interface = single_dielectric_box3d_rhs_adaptive(
    1.0, 1.0, 1.0, 4, vs, 1.0, 0.25, 1e-3, 4.0, 1.0, Float64)

rhs = rhs_dielectric_box3d_hybrid(interface, vs, 1.0, 1e-6)
```

When the same source is reused for many target batches or assemblies, build the field
once and reuse it. `PrecomputedVolumeField` stores a truncated spectral representation;
in-box targets are evaluated with a type-2 NUFFT and out-of-box targets with the FMM:

```julia
field = PrecomputedVolumeField(vs; tol = 1e-6)

phi  = volume_field_potential(field, targets)   # targets: 3×n  ->  length-n potential
grad = volume_field_gradient(field, targets)    #              ->  3×n gradient
rhs  = rhs_dielectric_box3d_field(interface, field, 1.0)
```

## Many right-hand sides and four-index integrals

For several sources sharing one interface, assemble and block-solve all right-hand sides
at once, then contract into the four-index Coulomb matrix `V[a, b]`:

```julia
using BoundaryIntegral

# vss::Vector{VolumeSource} on a shared interface that resolves all of them
# (e.g. built with multi_dielectric_box3d_rhs_adaptive)
sigma, stats = solve_dielectric_box3d_block(interface, vss; rtol = 1e-6)
V = four_index_matrix(interface, vss, sigma; lhs_tol = 1e-6, volume_tol = 1e-6)
```

## Visualization

Plotting helpers (`viz_2d`, `viz_3d`, `viz_3d_surface`, `viz_3d_interface_solution`,
`viz_3d_zslice`, `plot_campaign_geometry`) live in a Makie package extension. Install Makie
and a backend, then load the backend alongside `BoundaryIntegral`:

```julia
using Pkg
Pkg.add(["Makie", "CairoMakie"])
```

```julia
using CairoMakie
using BoundaryIntegral

interface = single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, 4.0, 1.0, Float64)
fig = viz_3d(interface)
```
