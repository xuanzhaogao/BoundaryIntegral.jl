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

# Flatten an (nx,ny,nz) grid array in the SAME order VolumeSource uses (ix outer, iz inner),
# so a column built this way is consistent with VolumeSource positions/weights.
function _flatten_grid_array(dg, A::AbstractArray{<:Real,3})
    nx, ny, nz = dg.nx, dg.ny, dg.nz
    v = Vector{Float64}(undef, nx * ny * nz)
    idx = 0
    @inbounds for ix in 1:nx, iy in 1:ny, iz in 1:nz
        idx += 1
        v[idx] = A[ix, iy, iz]
    end
    return v
end

# Read a datagrid once per path, caching it (the .xsf files are ~tens of MB each).
function _cached_xsf_grid!(cache::AbstractDict, path::AbstractString)
    haskey(cache, path) && return cache[path]
    _, dg = read_xsf(path)
    cache[path] = dg
    return dg
end

function pair_density_source(xsf_i::AbstractString, xsf_j::AbstractString; tol::Real = 0.0)
    _, dg_i = read_xsf(xsf_i)
    xsf_i == xsf_j && return VolumeSource(dg_i, dg_i.values .* dg_i.values; tol = tol)
    _, dg_j = read_xsf(xsf_j)
    return VolumeSource(dg_i, _pair_density_array(dg_i, dg_j); tol = tol)
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

# Datagrid for one orbital instance: the raw (cached) .xsf grid, circshifted by the orbital's
# lattice-image shift if any. Same grid (origin/vectors) — so all instances stay grid-compatible.
function _instance_grid(orb::OrbitalEntry, cache::AbstractDict)
    raw = _cached_xsf_grid!(cache, orb.xsf_path)
    orb.grid_shift == (0, 0, 0) && return raw
    return merge(raw, (; values = circshift(raw.values, orb.grid_shift)))
end

"""
    assemble_rhs_group(si, center_id; support_rtol=1e-6, grid_cache=Dict{String,Any}())

Assemble one center's batch of pair densities `rho_{i,j}` (j in the resolved group) on phi_i's
grid, as an `RHSGroup`. Each orbital's datagrid is read at most once (cached in `grid_cache`,
which may be shared across calls). The group is then truncated to its **union support**: points
where the envelope `sqrt(sum_k rho_k^2)` is below `support_rtol * (global max)` are dropped, the
SAME index set applied to every column so all densities stay on shared grid points (required for
nd-batched FMM). The threshold is relative to the GLOBAL envelope max, so significant pairs keep
their support while weakly-overlapping far pairs (tiny everywhere) keep few points. Pass
`support_rtol = 0` to keep the full grid.
"""
function assemble_rhs_group(si::SystemInput, center_id::Int;
        support_rtol::Real = 1e-6, grid_cache::AbstractDict = Dict{String, Any}())
    haskey(si.groups, center_id) || error("no group for center $center_id")
    js = si.groups[center_id]
    isempty(js) && error("group $center_id is empty")
    K = length(js)
    dg_i = _instance_grid(si.orbitals[center_id], grid_cache)

    # column 1 also gives the shared positions/weights (full grid, canonical ordering)
    base = VolumeSource(dg_i, _pair_density_array(dg_i, _instance_grid(si.orbitals[js[1]], grid_cache)))
    n = length(base.density)
    densities = Matrix{Float64}(undef, n, K)
    densities[:, 1] .= base.density
    for k in 2:K
        dg_j = _instance_grid(si.orbitals[js[k]], grid_cache)
        densities[:, k] .= _flatten_grid_array(dg_i, _pair_density_array(dg_i, dg_j))
    end

    # union-support truncation: keep points where the group envelope sqrt(sum_k rho_k^2) is
    # >= support_rtol * its global max. Relative to the GLOBAL max (not per-source): this keeps
    # the support of significant pairs (on-site, nearby off-site) while keeping few points for
    # weakly-overlapping far pairs (whose density is tiny everywhere) — a per-source-relative
    # threshold would instead keep ~the whole grid for such weak pairs (noise-floor support).
    keep = if support_rtol > 0
        env = vec(sqrt.(sum(abs2, densities; dims = 2)))
        m = maximum(env)
        m > 0 ? findall(>=(support_rtol * m), env) : collect(1:n)
    else
        collect(1:n)
    end
    return RHSGroup(center_id, copy(js), base.positions[:, keep], base.weights[keep], densities[keep, :])
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
    thresh::Float64,
) where {P <: FlatPanel{Float64, 3}}
    n_points = num_points(interface)
    K = length(vss)
    K == 0 && return zeros(Float64, n_points, 0)
    screened = [screened_volume_source(interface, vs, SharpScreening()) for vs in vss]

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
) where {P <: FlatPanel{Float64, 3}}
    op = batched_lhs_dielectric_box3d_fmm3d_corrected(interface, fmm_tol, up_tol, max_order)
    F = rhs_dielectric_box3d_fmm3d(interface, vss, fmm_tol)
    return Krylov.block_gmres(op, F; rtol = rtol, atol = atol, itmax = itmax)
end

# Split an RHSGroup's shared-grid density columns into the Vector{VolumeSource} core form.
function group_volume_sources(g::RHSGroup)
    return VolumeSource{Float64, 3}[
        VolumeSource(copy(g.positions), copy(g.weights), g.densities[:, k]) for k in 1:num_pairs(g)
    ]
end

"""
    solve_dielectric_box3d_group(si::SystemInput, center_id; support_rtol=si.solve.support_rtol)
    solve_dielectric_box3d_group(bie_path, center_id; ...)

High-level Steps 0–6 (all solver knobs from the `.bie` `BEGIN_SOLVE` block / `si.solve`):
assemble the center group's pair densities, build ONE shared interface refined on the group
envelope (rss of the K densities — one FMM/depth, not K) in the real `.xsf` coordinate frame,
and block-GMRES solve for the layer densities Σ (N×K).
Returns `(; sigma, interface, sources, group, stats, labels)`. Uses the `Vector{VolumeSource}`
core throughout.
"""
function solve_dielectric_box3d_group(si::SystemInput, center_id::Int;
        support_rtol::Real = si.solve.support_rtol)
    group = assemble_rhs_group(si, center_id; support_rtol = support_rtol)
    js = group.neighbor_ids
    labels = ["rho_$(center_id)_$(j)" for j in js]
    isempty(js) && return (; sigma = zeros(0, 0), interface = nothing,
        sources = VolumeSource{Float64, 3}[], group = group, stats = nothing, labels = labels)
    sp = si.solve
    sources = group_volume_sources(group)
    # Refine on a single ENVELOPE source (rss of the K densities) — one FMM/TKM evaluation per
    # depth instead of K. The envelope is significant wherever any source is, so it resolves the
    # combined features; the per-source vector method costs K× more and matters little here.
    interface = build_group_interface(si, group;
        n_quad = sp.n_quad, rhs_atol = sp.rhs_tol, l_ec = resolved_l_ec(si),
        eps_out = si.eps_out, max_depth = sp.max_depth)
    Σ, stats = solve_dielectric_box3d_block(interface, sources;
        fmm_tol = sp.lhs_tol, up_tol = sp.lhs_tol, max_order = sp.max_order, rtol = sp.gmres_rtol)
    return (; sigma = Σ, interface = interface, sources = sources, group = group,
            stats = stats, labels = labels)
end

solve_dielectric_box3d_group(bie_path::AbstractString, center_id::Int; kw...) =
    solve_dielectric_box3d_group(read_system_input(bie_path), center_id; kw...)

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

"""
    four_index_integrals(si_or_bie_path, center_id; kw...) -> (; V, labels, sigma, interface, sources, group, stats)

One-shot Steps 0–7 for a center group: solve for Σ, then evaluate the K×K four-index matrix V.
"""
function four_index_integrals(si::SystemInput, center_id::Int; kw...)
    sol = solve_dielectric_box3d_group(si, center_id; kw...)
    sp = si.solve
    V = isempty(sol.sources) ? zeros(Float64, 0, 0) :
        four_index_matrix(sol.interface, sol.sources, sol.sigma;
                          lhs_tol = sp.lhs_tol, volume_tol = sp.volume_tol)
    return (; V = V, labels = sol.labels, sigma = sol.sigma, interface = sol.interface,
            sources = sol.sources, group = sol.group, stats = sol.stats)
end

four_index_integrals(bie_path::AbstractString, center_id::Int; kw...) =
    four_index_integrals(read_system_input(bie_path), center_id; kw...)
