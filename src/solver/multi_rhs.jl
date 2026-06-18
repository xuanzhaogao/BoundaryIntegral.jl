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
# Product values phi_i .* phi_j as an (nx,ny,nz) array on phi_i's grid. Compatible grids ->
# direct pointwise product; otherwise resample phi_j onto phi_i's grid (trilinear, NaN->0).
function _pair_density_array(dg_i, dg_j)
    if dg_i === dg_j || datagrids_compatible(dg_i, dg_j)
        return dg_i.values .* dg_j.values
    end
    Minv_j = _datagrid_affine(dg_j)[3]
    prod = similar(dg_i.values)
    nx, ny, nz = dg_i.nx, dg_i.ny, dg_i.nz
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        p = grid_point(dg_i, ix, iy, iz)
        vj = _datagrid_trilinear_value(dg_j, Minv_j, p)
        prod[ix, iy, iz] = dg_i.values[ix, iy, iz] * (isnan(vj) ? 0.0 : vj)
    end
    return prod
end

function pair_density_source(xsf_i::AbstractString, xsf_j::AbstractString; tol::Real = 0.0)
    _, dg_i = read_xsf(xsf_i)
    xsf_i == xsf_j && return VolumeSource(dg_i, dg_i.values .* dg_i.values; tol = tol)
    _, dg_j = read_xsf(xsf_j)
    return VolumeSource(dg_i, _pair_density_array(dg_i, dg_j); tol = tol)
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
    # glue loops are threaded over i (n nodes >> K); each i touches disjoint columns/rows.
    # Needs Julia started with threads (julia -t N / JULIA_NUM_THREADS), independent of OMP.
    Threads.@threads for i in 1:n
        @inbounds begin
            wi = op.weights[i]
            for k in 1:K
                charges[k, i] = wi * X[i, k]
            end
        end
    end
    vals = lfmm3d(op.thresh, op.sources, charges = charges, pg = 2, nd = K)
    grad = reshape(vals.grad, K, 3, n)    # (K, 3, n); handles nd==1 singleton drop
    C = op.corrections * X                # n x K (sparse mat-mat)
    Threads.@threads for i in 1:n
        @inbounds begin
            n1 = op.norms[1, i]; n2 = op.norms[2, i]; n3 = op.norms[3, i]; di = op.diag[i]
            for k in 1:K
                gn = n1 * grad[k, 1, i] + n2 * grad[k, 2, i] + n3 * grad[k, 3, i]
                Y[i, k] = -gn / (4π) + C[i, k] + di * X[i, k]
            end
        end
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

# thin wrapper so tests can call block_gmres directly on the operator
function _block_gmres_solve(op::BatchedDielectricOperator, F::AbstractMatrix;
        rtol::Float64 = 1e-10, atol::Float64 = 0.0, itmax::Int = 500)
    Σ, stats = Krylov.block_gmres(op, Matrix{Float64}(F); rtol = rtol, atol = atol, itmax = itmax)
    return Σ, stats
end

# true iff every source is sampled on identical grid points (=> one FMM tree can serve all)
function _sources_share_positions(vss::AbstractVector)
    length(vss) <= 1 && return true
    p1 = vss[1].positions
    for k in 2:length(vss)
        (size(vss[k].positions) == size(p1) && vss[k].positions == p1) || return false
    end
    return true
end

"""
    rhs_dielectric_box3d_fmm3d(interface, vss::Vector{VolumeSource}, thresh) -> N×K matrix

Multi-source right-hand side: column k is the BIE RHS for source `vss[k]`, mirroring the
single-source `rhs_dielectric_box3d_fmm3d(interface, vs, thresh)` (each source screened via
the interface, `eps_src = 1`). When all (screened) sources share identical grid points, a
single `nd = K` batched FMM is used; otherwise the per-source method is looped.
"""
function rhs_dielectric_box3d_fmm3d(
    interface::DielectricInterface{P, Float64},
    vss::Vector{<:VolumeSource{Float64, 3}},
    thresh::Float64;
    screen_boxes::Union{Nothing, Vector{<:NamedTuple}} = nothing,
    screen_epses::Union{Nothing, Vector{Float64}} = nothing,
    screen_eps_out::Float64 = 1.0,
) where {P <: FlatPanel{Float64, 3}}
    n_points = num_points(interface)
    K = length(vss)
    K == 0 && return zeros(Float64, n_points, 0)
    # Default: interface-based screening (single eps region). With `screen_boxes` given,
    # screen each source box-based (multi-region) — the interface may span several eps
    # regions (e.g. a heterojunction substrate) where interface-based screening is undefined.
    screened = screen_boxes === nothing ?
        [screened_volume_source(interface, vs, SharpScreening()) for vs in vss] :
        [screened_volume_source(screen_boxes, screen_epses, screen_eps_out, vs, SharpScreening()) for vs in vss]

    if !_sources_share_positions(screened)
        F = Matrix{Float64}(undef, n_points, K)
        for k in 1:K
            F[:, k] = rhs_dielectric_box3d_fmm3d(interface, screened[k], 1.0, thresh)
        end
        return F
    end

    # shared grid -> one nd=K batched FMM (matches the per-source formula exactly)
    sources = screened[1].positions
    n = size(sources, 2)
    charges = Matrix{Float64}(undef, K, n)
    @inbounds for k in 1:K, s in 1:n
        charges[k, s] = screened[k].weights[s] * screened[k].density[s]
    end
    targets, normals = _interface_targets_normals(interface)
    vals = lfmm3d(thresh, sources, charges = charges, targets = targets, pgt = 2, nd = K)
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
    solve_dielectric_box3d_block(interface, vss::Vector{VolumeSource}; kw...) -> (Σ, stats)

Low-level multi-RHS solve on a prebuilt shared interface: builds the batched operator and the
N×K RHS, then block-GMRES solves A Σ = F for the layer densities Σ (N×K).
"""
function solve_dielectric_box3d_block(
    interface::DielectricInterface{P, Float64},
    vss::Vector{<:VolumeSource{Float64, 3}};
    fmm_tol::Float64 = 1e-9, up_tol::Float64 = 1e-9, max_order::Int = 8,
    rtol::Float64 = 1e-10, atol::Float64 = 0.0, itmax::Int = 500,
    screen_boxes::Union{Nothing, Vector{<:NamedTuple}} = nothing,
    screen_epses::Union{Nothing, Vector{Float64}} = nothing,
    screen_eps_out::Float64 = 1.0,
) where {P <: FlatPanel{Float64, 3}}
    op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, fmm_tol, up_tol, max_order)
    F = rhs_dielectric_box3d_fmm3d(interface, vss, fmm_tol;
        screen_boxes = screen_boxes, screen_epses = screen_epses, screen_eps_out = screen_eps_out)
    return Krylov.block_gmres(op, F; rtol = rtol, atol = atol, itmax = itmax)
end

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

