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
    build_group_interface(si, g; n_quad, rhs_atol, l_ec, eps_out=si.eps_out, max_depth=128)

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
