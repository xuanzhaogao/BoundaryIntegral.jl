"""
    PrecomputedVolumeField(vs; tol, kmax = nothing, margin_h = 5.0,
                           compute_pot = true, compute_grad = true)

Target-independent precomputed spectral representation of the free-space
Laplace potential (`1/(4π r)` convention) of a `VolumeSource`, for evaluation
at many target batches without redoing per-call setup.

Construction (once): fix the evaluation box B = source bounding box extended
by `margin_h * h` (`h` = mean source spacing); on the Fourier box determined
by B alone, run the type-1 NUFFT of all source charges, apply the truncated
Laplace kernel `TKM3D.truncated_laplace3d_hat`, and (optionally) materialize
the spectral gradient coefficients.

Evaluation (per batch): targets inside B use a type-2 NUFFT on the stored
coefficients — the FFT dimensions never change, so FFTW planning is reused;
targets outside B use an exact direct threaded sum (no FMM setup).

This removes the dominant cost of the RHS-adaptive mesh build, where the
target-dependent Fourier box previously forced a fresh type-1 NUFFT + FFT
plan at every refinement depth (~23 s per call on the production monolayer
density; measured 16x build speedup with max rel deviation 1.4e-5 and no
refinement-decision changes).
"""
struct PrecomputedVolumeField{T <: AbstractFloat}
    sources::Matrix{T}
    charges::Vector{T}
    lo::NTuple{3, T}
    hi::NTuple{3, T}
    center::NTuple{3, T}
    dks::NTuple{3, T}
    nmodes::NTuple{3, Int}
    kmax::T
    tol::T
    prefactor::T
    coeff::Union{Nothing, Array{Complex{T}, 3}}
    grad_coeff::Union{Nothing, Array{Complex{T}, 4}}
end

function PrecomputedVolumeField(
    vs::VolumeSource{T, 3};
    tol::Real,
    kmax::Union{Nothing, Real} = nothing,
    margin_h::Real = 5.0,
    compute_pot::Bool = true,
    compute_grad::Bool = true,
) where {T <: AbstractFloat}
    (compute_pot || compute_grad) || throw(ArgumentError("at least one of compute_pot or compute_grad must be true"))
    tolT = T(tol)
    tolT > zero(T) || throw(ArgumentError("tol must be positive"))
    sources, charges = _volume_source_fmm_sources(vs)
    h = _estimate_source_spacing(vs)
    km = isnothing(kmax) ? T(_estimate_tkm3dc_kmax(h)) : T(kmax)
    km > zero(T) || throw(ArgumentError("kmax must be positive"))

    m = T(margin_h) * h
    lo = ntuple(d -> minimum(view(sources, d, :)) - m, 3)
    hi = ntuple(d -> maximum(view(sources, d, :)) + m, 3)
    corners = T[lo[1] hi[1]; lo[2] hi[2]; lo[3] hi[3]]
    lengths, center = TKM3D.combined_box_geometry_3xn(sources, corners)
    Lbig = sqrt(sum(abs2, lengths))
    dks = ntuple(d -> prevfloat(T(2π) / (lengths[d] + Lbig)), 3)
    kx = TKM3D.centered_mode_axis(dks[1], km)
    ky = TKM3D.centered_mode_axis(dks[2], km)
    kz = TKM3D.centered_mode_axis(dks[3], km)
    nmodes = (length(kx), length(ky), length(kz))

    srcx = dks[1] .* (vec(view(sources, 1, :)) .- center[1])
    srcy = dks[2] .* (vec(view(sources, 2, :)) .- center[2])
    srcz = dks[3] .* (vec(view(sources, 3, :)) .- center[3])
    coeff0 = TKM3D.FINUFFT.nufft3d1(srcx, srcy, srcz, complex.(charges), -1, tolT, nmodes...)
    coeff = ndims(coeff0) == 4 ? dropdims(coeff0; dims = 4) : coeff0
    @inbounds for iz in eachindex(kz), iy in eachindex(ky), ix in eachindex(kx)
        k = sqrt(kx[ix]^2 + ky[iy]^2 + kz[iz]^2)
        coeff[ix, iy, iz] = k <= km ?
            coeff[ix, iy, iz] * TKM3D.truncated_laplace3d_hat(k, Lbig) :
            zero(eltype(coeff))
    end
    grad_coeff = compute_grad ? TKM3D._spectral_gradient_coeffs_3d(coeff, kx, ky, kz) : nothing
    prefactor = dks[1] * dks[2] * dks[3] / T(2π)^3
    return PrecomputedVolumeField{T}(
        sources, charges, lo, hi, (center[1], center[2], center[3]),
        dks, nmodes, km, tolT, prefactor,
        compute_pot ? coeff : nothing, grad_coeff)
end

@inline in_field_box(f::PrecomputedVolumeField, targets::AbstractMatrix, i::Integer) =
    (f.lo[1] <= targets[1, i] <= f.hi[1]) &&
    (f.lo[2] <= targets[2, i] <= f.hi[2]) &&
    (f.lo[3] <= targets[3, i] <= f.hi[3])

function _field_scaled_targets(f::PrecomputedVolumeField{T}, targets, idxs) where {T}
    txn = T[f.dks[1] * (targets[1, i] - f.center[1]) for i in idxs]
    tyn = T[f.dks[2] * (targets[2, i] - f.center[2]) for i in idxs]
    tzn = T[f.dks[3] * (targets[3, i] - f.center[3]) for i in idxs]
    return txn, tyn, tzn
end

function _direct_potential!(out::AbstractVector{T}, sources, charges, targets, idxs) where {T}
    isempty(idxs) && return out
    sx = sources[1, :]; sy = sources[2, :]; sz = sources[3, :]
    Threads.@threads for ii in eachindex(idxs)
        i = idxs[ii]
        x, y, z = targets[1, i], targets[2, i], targets[3, i]
        acc = zero(T)
        @inbounds @simd for j in eachindex(charges)
            dx = x - sx[j]; dy = y - sy[j]; dz = z - sz[j]
            acc += charges[j] / sqrt(dx * dx + dy * dy + dz * dz)
        end
        out[i] = acc / (4 * T(π))
    end
    return out
end

function _direct_gradient!(out::AbstractMatrix{T}, sources, charges, targets, idxs) where {T}
    isempty(idxs) && return out
    sx = sources[1, :]; sy = sources[2, :]; sz = sources[3, :]
    Threads.@threads for ii in eachindex(idxs)
        i = idxs[ii]
        x, y, z = targets[1, i], targets[2, i], targets[3, i]
        gx = zero(T); gy = zero(T); gz = zero(T)
        @inbounds @simd for j in eachindex(charges)
            dx = x - sx[j]; dy = y - sy[j]; dz = z - sz[j]
            r2 = dx * dx + dy * dy + dz * dz
            s = charges[j] / (r2 * sqrt(r2))
            gx -= s * dx; gy -= s * dy; gz -= s * dz
        end
        out[1, i] = gx / (4 * T(π))
        out[2, i] = gy / (4 * T(π))
        out[3, i] = gz / (4 * T(π))
    end
    return out
end

"""
    volume_field_potential(field, targets) -> Vector

Potential of the precomputed field at `targets` (3 x n), free-space
`1/(4π r)` normalization. In-box targets via type-2 NUFFT, out-of-box via
direct threaded summation.
"""
function volume_field_potential(f::PrecomputedVolumeField{T}, targets::AbstractMatrix{<:Real}) where {T}
    size(targets, 1) == 3 || throw(ArgumentError("targets must have shape (3, n)"))
    f.coeff === nothing && throw(ArgumentError("field was built with compute_pot = false"))
    trg = Matrix{T}(targets)
    n = size(trg, 2)
    out = Vector{T}(undef, n)
    inb = [in_field_box(f, trg, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    if !isempty(bidx)
        txn, tyn, tzn = _field_scaled_targets(f, trg, bidx)
        vals = TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, f.coeff)
        for (k, i) in enumerate(bidx)
            out[i] = f.prefactor * real(vals[k])
        end
    end
    _direct_potential!(out, f.sources, f.charges, trg, oidx)
    return out
end

"""
    volume_field_gradient(field, targets) -> 3 x n Matrix

Gradient of the precomputed field at `targets` (3 x n), free-space
`1/(4π r)` normalization.
"""
function volume_field_gradient(f::PrecomputedVolumeField{T}, targets::AbstractMatrix{<:Real}) where {T}
    size(targets, 1) == 3 || throw(ArgumentError("targets must have shape (3, n)"))
    f.grad_coeff === nothing && throw(ArgumentError("field was built with compute_grad = false"))
    trg = Matrix{T}(targets)
    n = size(trg, 2)
    out = Matrix{T}(undef, 3, n)
    inb = [in_field_box(f, trg, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    if !isempty(bidx)
        txn, tyn, tzn = _field_scaled_targets(f, trg, bidx)
        vals = TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, f.grad_coeff)
        for (k, i) in enumerate(bidx)
            out[1, i] = f.prefactor * real(vals[k, 1])
            out[2, i] = f.prefactor * real(vals[k, 2])
            out[3, i] = f.prefactor * real(vals[k, 3])
        end
    end
    _direct_gradient!(out, f.sources, f.charges, trg, oidx)
    return out
end

"""
    _rhs_volume_targets_field(field, targets, normals, eps_src) -> Vector

Dielectric-interface RHS values `-(n · ∇φ) / eps_src` at `targets` from a
precomputed field (free-space normalization, matching the TKM branch of
`_rhs_volume_targets_hybrid`). Replaces the per-call KDTree classify +
lfmm3d/ltkm3dc pair with one field evaluation.
"""
function _rhs_volume_targets_field(
    field::PrecomputedVolumeField{T},
    targets::Matrix{T},
    normals::Matrix{T},
    eps_src::T,
) where {T}
    n = size(targets, 2)
    @assert size(normals, 2) == n
    grad = volume_field_gradient(field, targets)
    rhs_vals = Vector{T}(undef, n)
    @inbounds for i in 1:n
        rhs_vals[i] = -(normals[1, i] * grad[1, i] + normals[2, i] * grad[2, i] +
                        normals[3, i] * grad[3, i]) / eps_src
    end
    return rhs_vals
end
