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

# Batched RHS for a whole center group on a shared grid.
# Returns an N x K matrix F whose column k is f_{i, neighbor_ids[k]} = -∂n u_inc[rho_ij].
# Placed in multi_rhs.jl (not dielectric_box3d.jl) because the signature references
# SystemInput/RHSGroup, which are defined here / in system_input.jl and are not yet
# bound at the point dielectric_box3d.jl is included (parse-time resolution).
function rhs_dielectric_box3d_fmm3d_batched(
    interface::DielectricInterface{P, Float64},
    si::SystemInput,
    group::RHSGroup,
    thresh::Float64,
) where {P <: AbstractPanel}
    n = size(group.positions, 2)
    K = num_pairs(group)
    n_points = num_points(interface)
    n == 0 && return zeros(Float64, n_points, K)

    # Per-point screening factor 1/eps_local (identical across columns -> nd-batchable).
    # Box containment must match the reference screening (_screened_volume_density_multibox):
    # same _point_in_box helper and the same boundary tolerance, so a point on/near a box
    # face is classified identically and the batched RHS reproduces the single-RHS path.
    box_tol = sqrt(eps(Float64)) * maximum(b -> max(abs(b.Lx), abs(b.Ly), abs(b.Lz)), si.boxes)
    inv_eps = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        pos = (group.positions[1, s], group.positions[2, s], group.positions[3, s])
        eps_local = si.eps_out
        for b in eachindex(si.boxes)
            box = si.boxes[b]
            lo = (box.center[1] - box.Lx/2, box.center[2] - box.Ly/2, box.center[3] - box.Lz/2)
            hi = (box.center[1] + box.Lx/2, box.center[2] + box.Ly/2, box.center[3] + box.Lz/2)
            if _point_in_box(pos, lo, hi, box_tol)
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

    vals = lfmm3d(thresh, group.positions, charges = charges, targets = targets, pgt = 2, nd = K)
    # FMM3D returns gradtarg as (nd, 3, n_points) for nd > 1, but drops the leading
    # singleton dim to (3, n_points) when nd == 1. Reshape to a uniform (K, 3, n_points).
    grad = reshape(vals.gradtarg, K, 3, n_points)
    F = Matrix{Float64}(undef, n_points, K)
    @inbounds for k in 1:K, i in 1:n_points
        F[i, k] = (normals[1, i] * grad[k, 1, i] +
                   normals[2, i] * grad[k, 2, i] +
                   normals[3, i] * grad[k, 3, i]) / (4π)
    end
    return F
end

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
    vals = lfmm3d(op.thresh, op.sources, charges = charges, pg = 2, nd = K)
    grad = reshape(vals.grad, K, 3, n)    # (K, 3, n); handles nd==1 singleton drop
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
