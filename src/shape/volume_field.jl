"""
    PrecomputedVolumeField(vs; tol, kmax = nothing, margin_h = 5.0,
                           compute_pot = true, compute_grad = true,
                           cache_fft = false, cache_fft_pad = 1.25)

Target-independent precomputed spectral representation of the free-space
Laplace potential (`1/(4π r)` convention) of a `VolumeSource`, for evaluation
at many target batches without redoing per-call setup.

Construction (once): fix the evaluation box B = source bounding box extended
by `margin_h * h` (`h` = mean source spacing); on the Fourier box determined
by B alone, run the type-1 NUFFT of all source charges, apply the truncated
Laplace kernel `TKM3D.truncated_laplace3d_hat`, and (optionally) materialize
the spectral gradient coefficients.

Evaluation (per batch): targets inside B use a type-2 NUFFT on the stored
coefficients — near the source support this spectral evaluation is the only
accurate option (point-charge summation has O(1%) quadrature error there);
the FFT dimensions never change, so FFTW planning is reused. Targets outside
B are well separated from the support and are evaluated with the FMM
(`lfmm3d` at the field tolerance) — equivalent to the exact point sum within
tolerance and asymptotically scalable in the batch size.

This removes the dominant cost of the RHS-adaptive mesh build, where the
target-dependent Fourier box previously forced a fresh type-1 NUFFT + FFT
plan at every refinement depth (~23 s per call on the production monolayer
density; measured 16x build speedup with max rel deviation 1.4e-5 and no
refinement-decision changes).

Memory: `coeff` holds `prod(nmodes)` complex doubles and `grad_coeff` three
times that (≈1 GB and ≈3 GB at the production kmax); construction with
`compute_grad = true` transiently holds both even if `compute_pot = false`.
Use the `compute_pot`/`compute_grad` flags to store only what the consumer
needs.

Cached-FFT mode (`cache_fft = true`, experimental): the standard in-box path
re-runs the FFT of the full coefficient grid inside FINUFFT on EVERY type-2
exec (~0.4 s at production scale, independent of the target count). With
`cache_fft = true` that FFT is done once at construction: the spectral
coefficients are divided by the FINUFFT spreading-kernel Fourier factors
(pre-deconvolution: the interpolation kernel's smoothing is corrected per
mode up front), zero-padded onto a fine grid of `cache_fft_pad * nmodes`
per axis, and inverse-FFT'd; per evaluation a native threaded ES-kernel
interpolation reads the stored fine grid (cost ∝ targets, not modes —
equivalent to FINUFFT's `spreadinterponly` type-2 exec, see
`_field_interponly_type2` for why it is not routed through finufft itself).
The coefficient arrays are dropped and replaced by the fine grids
(`cache_fft_pad^3` ≈ 2x their size at the default 1.25). Accuracy relies on
the spectrum having decayed to ~tol at `kmax` (guaranteed by the kmax
estimate); the padding keeps the active modes away from the fine-grid
Nyquist, where the upsampfac≈1 interpolation kernel loses accuracy —
validated at ~5e-9 max relative deviation from the standard path (pad 1.25;
pad 1.0 degrades the gradient to ~2e-6).
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
    # cache_fft mode: kernel-corrected fine-grid values for interp-only type-2
    nfdim::Union{Nothing, NTuple{3, Int}}
    pot_grid::Union{Nothing, Array{Complex{T}, 3}}
    grad_grid::Union{Nothing, Array{Complex{T}, 4}}
end

function PrecomputedVolumeField(
    vs::VolumeSource{T, 3};
    tol::Real,
    kmax::Union{Nothing, Real} = nothing,
    margin_h::Real = 5.0,
    compute_pot::Bool = true,
    compute_grad::Bool = true,
    cache_fft::Bool = false,
    cache_fft_pad::Real = 1.25,
) where {T <: AbstractFloat}
    (compute_pot || compute_grad) || throw(ArgumentError("at least one of compute_pot or compute_grad must be true"))
    cache_fft_pad >= 1 || throw(ArgumentError("cache_fft_pad must be >= 1"))
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
    nfdim = nothing; pot_grid = nothing; grad_grid = nothing
    if cache_fft
        nfdim, pot_grid, grad_grid =
            _field_cache_fft_grids(coeff, grad_coeff, tolT, T(cache_fft_pad), compute_pot)
    end
    return PrecomputedVolumeField{T}(
        sources, charges, lo, hi, (center[1], center[2], center[3]),
        dks, nmodes, km, tolT, prefactor,
        (compute_pot && !cache_fft) ? coeff : nothing,
        cache_fft ? nothing : grad_coeff,
        nfdim, pot_grid, grad_grid)
end

# --- cache_fft mode internals -----------------------------------------------
# Mirrors TKM3D's discrete spread-only conventions (_ltkm3dd_eval_spreadonly,
# dir-2 leg): centered-mode coefficients are divided by the spreading-kernel
# Fourier factors phi (the deconvolution FINUFFT skips in spreadinterponly
# mode), placed in FFT ordering on the fine grid, and bfft'd (iflag = +1).
# A spreadinterponly type-2 exec on the result is then exactly equivalent to
# the standard type-2 on the original coefficients, provided the spectrum has
# decayed at the mode-box edge (see the struct docstring).

function _field_cache_fft_grids(
    coeff::Array{Complex{T}, 3},
    grad_coeff::Union{Nothing, Array{Complex{T}, 4}},
    tol::T,
    pad::T,
    compute_pot::Bool,
) where {T <: AbstractFloat}
    nmodes = size(coeff)
    sigma = TKM3D._ltkm3dd_spreadonly_upsampfac(nmodes, Float64(tol))
    params = TKM3D._ltkm3dd_spreadonly_kernel_params(Float64(tol), sigma; kerformula = 1)
    hc = TKM3D._ltkm3dd_spreadonly_horner_coeffs(params)
    # padded relative to the TKM rule: keeps the active modes away from the
    # fine-grid Nyquist, where the upsampfac≈1 kernel correction degrades
    nfdim = ntuple(d -> TKM3D._ltkm3dd_spreadonly_next235even(
        max(ceil(Int, Float64(pad) * params.upsampfac * nmodes[d]), 2 * params.nspread)), 3)
    phi1 = T.(TKM3D._ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[1], params, hc))
    phi2 = T.(TKM3D._ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[2], params, hc))
    phi3 = T.(TKM3D._ltkm3dd_spreadonly_onedim_fseries_kernel(nfdim[3], params, hc))
    pot_grid = nothing
    if compute_pot
        fw = TKM3D._ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(coeff, nfdim, phi1, phi2, phi3)
        pot_grid = TKM3D.FFTW.bfft(fw)
    end
    grad_grid = nothing
    if grad_coeff !== nothing
        grad_grid = Array{Complex{T}, 4}(undef, nfdim..., 3)
        for d in 1:3
            fw = TKM3D._ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(
                view(grad_coeff, :, :, :, d), nfdim, phi1, phi2, phi3)
            grad_grid[:, :, :, d] = TKM3D.FFTW.bfft(fw)
        end
    end
    return nfdim, pot_grid, grad_grid
end

# Interp-only type-2 on the stored fine grid; equivalent to
# TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, coeff) of the original
# coefficients.
#
# The interpolation is done natively (threaded Julia) rather than through a
# FINUFFT spreadinterponly plan: finufft 2.5.x's execute allocates and
# value-initializes its nf-sized internal workspace on EVERY exec even though
# spreadinterponly never touches it (finufft_core.cpp, fwBatch_), a measured
# ~6.4 ns/fine-grid-pt floor that would cost ~0.8 s per exec at production
# scale — more than the FFT this mode exists to avoid. The native interp is
# exactly FINUFFT's: same ES kernel piecewise polynomials (shared with the
# phi correction above), same fold u = frac(x/2π + 1/2)·nf — the half-period
# offset is compensated by the (-1)^k sign FINUFFT's kernel Fourier series
# bakes into phi (the `-exp` factor in _ltkm3dd_spreadonly_onedim_fseries_
# kernel). Validated against finufft's spreadinterponly exec to 1.4e-8.

# ES-kernel stencil for one coordinate: grid indices (1-based, periodic) and
# kernel weights for the ns fine-grid points supporting a target at x.
function _field_interp_stencil!(
    idx::Vector{Int},
    w::Vector{T},
    x::T,
    nf::Int,
    ns::Int,
    hc::Matrix{Float64},
) where {T <: AbstractFloat}
    u = mod(Float64(x) / (2π) + 0.5, 1.0) * nf   # FINUFFT fold: grid units in [0, nf)
    istart = ceil(Int, u - ns / 2)
    x1 = istart - u                              # in [-ns/2, -ns/2+1)
    @inbounds for l in 0:(ns - 1)
        idx[l + 1] = mod(istart + l, nf) + 1
        w[l + 1] = T(TKM3D._ltkm3dd_spreadonly_evaluate_kernel_runtime(x1 + l, hc, ns))
    end
    return nothing
end

function _field_interp3d!(
    out::AbstractVector{Complex{T}},
    grid::AbstractArray{Complex{T}, 3},
    txn::Vector{T},
    tyn::Vector{T},
    tzn::Vector{T},
    ns::Int,
    hc::Matrix{Float64},
) where {T <: AbstractFloat}
    nf1, nf2, nf3 = size(grid)
    n = length(txn)
    nch = max(1, min(Threads.nthreads(), n))
    Threads.@threads for c in 1:nch
        i1 = Vector{Int}(undef, ns); w1 = Vector{T}(undef, ns)
        i2 = Vector{Int}(undef, ns); w2 = Vector{T}(undef, ns)
        i3 = Vector{Int}(undef, ns); w3 = Vector{T}(undef, ns)
        @inbounds for j in (div((c - 1) * n, nch) + 1):div(c * n, nch)
            _field_interp_stencil!(i1, w1, txn[j], nf1, ns, hc)
            _field_interp_stencil!(i2, w2, tyn[j], nf2, ns, hc)
            _field_interp_stencil!(i3, w3, tzn[j], nf3, ns, hc)
            acc = zero(Complex{T})
            for l3 in 1:ns
                iz = i3[l3]
                accz = zero(Complex{T})
                for l2 in 1:ns
                    iy = i2[l2]
                    s = zero(Complex{T})
                    @simd for l1 in 1:ns
                        s += grid[i1[l1], iy, iz] * w1[l1]
                    end
                    accz += s * w2[l2]
                end
                acc += accz * w3[l3]
            end
            out[j] = acc
        end
    end
    return out
end

function _field_interponly_type2(
    f::PrecomputedVolumeField{T},
    txn::Vector{T},
    tyn::Vector{T},
    tzn::Vector{T},
    grid::Array{Complex{T}, N},
) where {T <: AbstractFloat, N}
    sigma = TKM3D._ltkm3dd_spreadonly_upsampfac(f.nfdim, Float64(f.tol))
    params = TKM3D._ltkm3dd_spreadonly_kernel_params(Float64(f.tol), sigma; kerformula = 1)
    hc = TKM3D._ltkm3dd_spreadonly_horner_coeffs(params)
    ntrans = N == 3 ? 1 : size(grid, 4)
    out = Matrix{Complex{T}}(undef, length(txn), ntrans)
    for d in 1:ntrans
        g = N == 3 ? grid : view(grid, :, :, :, d)
        _field_interp3d!(view(out, :, d), g, txn, tyn, tzn, params.nspread, hc)
    end
    return out
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


"""
    volume_field_potential(field, targets) -> Vector

Potential of the precomputed field at `targets` (3 x n), free-space
`1/(4π r)` normalization. In-box targets via type-2 NUFFT, out-of-box via
FMM at the field tolerance.
"""
function volume_field_potential(f::PrecomputedVolumeField{T}, targets::AbstractMatrix{<:Real}) where {T}
    size(targets, 1) == 3 || throw(ArgumentError("targets must have shape (3, n)"))
    (f.coeff === nothing && f.pot_grid === nothing) &&
        throw(ArgumentError("field was built with compute_pot = false"))
    trg = Matrix{T}(targets)
    n = size(trg, 2)
    out = Vector{T}(undef, n)
    inb = [in_field_box(f, trg, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    if !isempty(bidx)
        txn, tyn, tzn = _field_scaled_targets(f, trg, bidx)
        vals = f.pot_grid !== nothing ?
            vec(_field_interponly_type2(f, txn, tyn, tzn, f.pot_grid)) :
            TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, f.coeff)
        for (k, i) in enumerate(bidx)
            out[i] = f.prefactor * real(vals[k])
        end
    end
    if !isempty(oidx)
        vals = lfmm3d(f.tol, f.sources; charges = f.charges, targets = trg[:, oidx], pgt = 1)
        for (k, i) in enumerate(oidx)
            out[i] = vals.pottarg[k] / (4 * T(π))   # FMM3D uses the 1/r kernel
        end
    end
    return out
end

"""
    volume_field_gradient(field, targets) -> 3 x n Matrix

Gradient of the precomputed field at `targets` (3 x n), free-space
`1/(4π r)` normalization. In-box targets via type-2 NUFFT, out-of-box via
FMM at the field tolerance.
"""
function volume_field_gradient(f::PrecomputedVolumeField{T}, targets::AbstractMatrix{<:Real}) where {T}
    size(targets, 1) == 3 || throw(ArgumentError("targets must have shape (3, n)"))
    (f.grad_coeff === nothing && f.grad_grid === nothing) &&
        throw(ArgumentError("field was built with compute_grad = false"))
    trg = Matrix{T}(targets)
    n = size(trg, 2)
    out = Matrix{T}(undef, 3, n)
    inb = [in_field_box(f, trg, i) for i in 1:n]
    bidx = findall(inb); oidx = findall(!, inb)
    if !isempty(bidx)
        txn, tyn, tzn = _field_scaled_targets(f, trg, bidx)
        vals = f.grad_grid !== nothing ?
            _field_interponly_type2(f, txn, tyn, tzn, f.grad_grid) :
            TKM3D._finufft_type2_eval_3d(txn, tyn, tzn, 1, f.tol, f.grad_coeff)
        for (k, i) in enumerate(bidx)
            out[1, i] = f.prefactor * real(vals[k, 1])
            out[2, i] = f.prefactor * real(vals[k, 2])
            out[3, i] = f.prefactor * real(vals[k, 3])
        end
    end
    if !isempty(oidx)
        vals = lfmm3d(f.tol, f.sources; charges = f.charges, targets = trg[:, oidx], pgt = 2)
        for (k, i) in enumerate(oidx)
            out[1, i] = vals.gradtarg[1, k] / (4 * T(π))   # FMM3D uses the 1/r kernel
            out[2, i] = vals.gradtarg[2, k] / (4 * T(π))
            out[3, i] = vals.gradtarg[3, k] / (4 * T(π))
        end
    end
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
    @assert size(normals, 1) == 3
    grad = volume_field_gradient(field, targets)
    rhs_vals = Vector{T}(undef, n)
    @inbounds for i in 1:n
        rhs_vals[i] = _rhs_from_grad(view(normals, :, i), view(grad, :, i), eps_src, one(T))
    end
    return rhs_vals
end
