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
    Minv_j = _datagrid_affine(dg_j)[3]          # inverse cell matrix, computed once
    prod = similar(dg_i.values)
    nx, ny, nz = dg_i.nx, dg_i.ny, dg_i.nz
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        p = grid_point(dg_i, ix, iy, iz)
        vj = _datagrid_trilinear_value(dg_j, Minv_j, p)
        prod[ix, iy, iz] = dg_i.values[ix, iy, iz] * (isnan(vj) ? 0.0 : vj)
    end
    return VolumeSource(dg_i, prod; tol = tol)
end
