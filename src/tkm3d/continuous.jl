function _ltkm3dc_mean_positive_gap(values::AbstractVector{<:Real})
    sorted = sort!(collect(Float64.(values)))
    positive_sum = 0.0
    positive_count = 0

    for i in 1:(length(sorted) - 1)
        gap = sorted[i + 1] - sorted[i]
        if gap > 0.0
            positive_sum += gap
            positive_count += 1
        end
    end

    positive_count > 0 || throw(ArgumentError("cannot estimate source spacing from degenerate coordinates; provide kmax explicitly"))
    return positive_sum / positive_count
end

function _ltkm3dc_estimate_kmax(sources::AbstractMatrix{<:Real})
    src = as_3xn(sources)
    dx = _ltkm3dc_mean_positive_gap(vec(view(src, 1, :)))
    dy = _ltkm3dc_mean_positive_gap(vec(view(src, 2, :)))
    dz = _ltkm3dc_mean_positive_gap(vec(view(src, 3, :)))
    return min(pi / dx, pi / dy, pi / dz)
end

function _ltkm3dc_source_nyquist_geometry(sources::AbstractMatrix{<:Real})
    src = as_3xn(sources)
    mins = vec(minimum(src; dims = 2))
    maxs = vec(maximum(src; dims = 2))
    hx = _ltkm3dc_mean_positive_gap(vec(view(src, 1, :)))
    hy = _ltkm3dc_mean_positive_gap(vec(view(src, 2, :)))
    hz = _ltkm3dc_mean_positive_gap(vec(view(src, 3, :)))

    spacings = (Float64(hx), Float64(hy), Float64(hz))
    lengths = (
        Float64(maxs[1] - mins[1] + hx),
        Float64(maxs[2] - mins[2] + hy),
        Float64(maxs[3] - mins[3] + hz),
    )
    Δk = (
        Float64(2π / lengths[1]),
        Float64(2π / lengths[2]),
        Float64(2π / lengths[3]),
    )
    axis_nyquist = (
        Float64(pi / hx),
        Float64(pi / hy),
        Float64(pi / hz),
    )
    return mins, spacings, lengths, Δk, axis_nyquist
end

"""
    estimate_kcut3dc(sources; charges, tol, eps=1e-12)

Estimate the smallest radial spectral cutoff `kcut` such that the relative
pointwise tail of the type-1 NUFFT spectrum satisfies

`max_{|q| > kcut} |F(q)| / max_q |F(q)| < tol`

using a Nyquist-limited anisotropic mode box inferred from the average positive
source-coordinate gaps on each axis.
"""
function estimate_kcut3dc(
    sources::AbstractMatrix{<:Real};
    charges,
    tol::Real,
    eps::Real = 1e-12,
)
    0.0 < tol < 1.0 || throw(ArgumentError("tol must satisfy 0 < tol < 1"))
    eps > 0 || throw(ArgumentError("eps must be positive"))

    src = as_3xn(sources)
    q = Vector{float(eltype(charges))}(charges)
    length(q) == size(src, 2) || throw(ArgumentError("number of charges must match number of sources"))

    mins, _, lengths, Δk, axis_nyquist = _ltkm3dc_source_nyquist_geometry(src)
    T = promote_type(Float64, eltype(src), eltype(q), typeof(eps))

    kx = centered_mode_axis(T(Δk[1]), T(axis_nyquist[1]))
    ky = centered_mode_axis(T(Δk[2]), T(axis_nyquist[2]))
    kz = centered_mode_axis(T(Δk[3]), T(axis_nyquist[3]))
    nmodes = (length(kx), length(ky), length(kz))

    srcx = T(Δk[1]) .* (vec(view(src, 1, :)) .- T(mins[1]))
    srcy = T(Δk[2]) .* (vec(view(src, 2, :)) .- T(mins[2]))
    srcz = T(Δk[3]) .* (vec(view(src, 3, :)) .- T(mins[3]))

    coeff = nufft3d1(srcx, srcy, srcz, complex.(T.(q)), -1, T(eps), nmodes...)
    if ndims(coeff) == 4 && size(coeff, 4) == 1
        coeff = dropdims(coeff; dims = 4)
    end

    radii, magnitudes = _spectral_radii_and_magnitudes(coeff, kx, ky, kz)
    kcut = T(_smallest_kcut_from_tail(radii, magnitudes, tol))
    tail_ratio = T(_pointwise_tail_ratio(radii, magnitudes, kcut))
    max_coeff = T(maximum(magnitudes))
    kmax_nyquist = T(maximum(axis_nyquist))

    return KCut3DCResult(
        kcut,
        kmax_nyquist,
        (T(axis_nyquist[1]), T(axis_nyquist[2]), T(axis_nyquist[3])),
        max_coeff,
        tail_ratio,
        nmodes,
        (T(Δk[1]), T(Δk[2]), T(Δk[3])),
    )
end

function _ltkm3dc_eval(
    sources::AbstractMatrix{<:Real},
    charges::AbstractVector{<:Real},
    targets::AbstractMatrix{<:Real},
    kmax::Real,
    eps::Real;
    compute_pot::Bool = true,
    compute_grad::Bool = false,
    nthreads::Integer = 0,
)
    kmax > 0 || throw(ArgumentError("kmax must be positive"))
    eps > 0 || throw(ArgumentError("eps must be positive"))

    src0 = as_3xn(sources)
    trg0 = as_3xn(targets)
    length(charges) == size(src0, 2) || throw(ArgumentError("number of charges must match number of sources"))

    T = promote_type(Float64, eltype(src0), eltype(trg0), eltype(charges), typeof(kmax), typeof(eps))
    src = Matrix{T}(src0)
    trg = Matrix{T}(trg0)
    q = Vector{T}(charges)

    lengths, center = combined_box_geometry_3xn(src, trg)
    l_x, l_y, l_z = lengths
    cx, cy, cz = center

    L = sqrt(l_x^2 + l_y^2 + l_z^2)
    L > zero(T) || throw(ArgumentError("source/target box must have positive extent"))

    Δk_x = prevfloat(T(2π) / (l_x + L))
    Δk_y = prevfloat(T(2π) / (l_y + L))
    Δk_z = prevfloat(T(2π) / (l_z + L))

    kx = centered_mode_axis(Δk_x, T(kmax))
    ky = centered_mode_axis(Δk_y, T(kmax))
    kz = centered_mode_axis(Δk_z, T(kmax))

    srcx = Δk_x .* (vec(view(src, 1, :)) .- cx)
    srcy = Δk_y .* (vec(view(src, 2, :)) .- cy)
    srcz = Δk_z .* (vec(view(src, 3, :)) .- cz)
    trgx = Δk_x .* (vec(view(trg, 1, :)) .- cx)
    trgy = Δk_y .* (vec(view(trg, 2, :)) .- cy)
    trgz = Δk_z .* (vec(view(trg, 3, :)) .- cz)

    coeff = nufft3d1(srcx, srcy, srcz, complex.(q), -1, T(eps), length(kx), length(ky), length(kz); nthreads = nthreads)
    if ndims(coeff) == 4 && size(coeff, 4) == 1
        coeff = dropdims(coeff; dims = 4)
    end

    @inbounds for iz in eachindex(kz), iy in eachindex(ky), ix in eachindex(kx)
        k = sqrt(kx[ix]^2 + ky[iy]^2 + kz[iz]^2)
        if k <= T(kmax)
            coeff[ix, iy, iz] *= truncated_laplace3d_hat(k, L)
        else
            coeff[ix, iy, iz] = zero(eltype(coeff))
        end
    end

    prefactor = (Δk_x * Δk_y * Δk_z) / (T(2π)^3)
    pot = nothing
    grad = nothing

    if compute_pot || compute_grad
        plan = _finufft_make_type2_plan_3d(trgx, trgy, trgz, 1, T(eps), (length(kx), length(ky), length(kz)), 1, T; nthreads = nthreads)
        try
            if compute_pot
                pot_complex = _finufft_type2_exec_3d(plan, coeff)
                pot = prefactor .* real.(pot_complex)
            end

            if compute_grad
                grad_coeff = _spectral_gradient_coeffs_3d(coeff, kx, ky, kz)
                FINUFFT.finufft_destroy!(plan)
                plan = nothing
                plan = _finufft_make_type2_plan_3d(trgx, trgy, trgz, 1, T(eps), (length(kx), length(ky), length(kz)), 3, T; nthreads = nthreads)
                grad_complex = _finufft_type2_exec_3d(plan, grad_coeff)
                grad = prefactor .* permutedims(real.(grad_complex))
            end
        finally
            !isnothing(plan) && FINUFFT.finufft_destroy!(plan)
        end
    end

    return pot, grad
end

# FINUFFT thread count for the internal type-1/type-2 transforms. FINUFFT's default FFTW
# planner (FFTW_ESTIMATE) picks a pathological multithreaded plan for the upsampled mode
# grids at high thread counts (measured: one type-1 on a 413x375x325 grid is 0.7 s at 16
# threads but 27.7 s at 96; a good plan execs in ~0.5 s at ANY thread count). Capping
# FINUFFT here loses no FFT throughput and leaves the FMM (lfmm3d, a separate library) on
# the full OpenMP pool. Override via TKM3D_FINUFFT_NTHREADS (0 = FINUFFT's omp default).
_ltkm_default_nthreads() = (v = get(ENV, "TKM3D_FINUFFT_NTHREADS", ""); isempty(v) ? 16 : parse(Int, v))

"""
    ltkm3dc(eps, sources; charges, targets=nothing, pg=0, pgt=0, kmax=nothing,
            nthreads=_ltkm_default_nthreads())

Continuous free-space Laplace TKM evaluator with arbitrary source points and
preweighted quadrature masses. `nthreads` caps the internal FINUFFT transforms
(default 16, or env `TKM3D_FINUFFT_NTHREADS`) to avoid the FFTW high-thread-count plan
pathology; the FMM path is unaffected.
"""
function ltkm3dc(
    eps::Real,
    sources::AbstractMatrix{<:Real};
    charges,
    targets::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
    pg::Integer = 0,
    pgt::Integer = 0,
    kmax::Union{Nothing, Real} = nothing,
    nthreads::Integer = _ltkm_default_nthreads(),
)
    pg in (0, 1, 2) || throw(ArgumentError("ltkm3dc currently supports only pg = 0, 1, or 2"))
    pgt in (0, 1, 2) || throw(ArgumentError("ltkm3dc currently supports only pgt = 0, 1, or 2"))
    eps > 0 || throw(ArgumentError("eps must be positive"))

    src = as_3xn(sources)
    q = Vector{float(eltype(charges))}(charges)
    length(q) == size(src, 2) || throw(ArgumentError("number of charges must match number of sources"))

    resolved_kmax = if isnothing(kmax)
        cut = estimate_kcut3dc(src; charges = q, tol = eps, eps = eps)
        max(Float64(cut.kcut), Float64(minimum(cut.Δk)))
    else
        Float64(kmax)
    end
    resolved_kmax > 0 || throw(ArgumentError("kmax must be positive"))

    pot = nothing
    grad = nothing
    pottarg = nothing
    gradtarg = nothing

    if pg > 0
        pot, grad = _ltkm3dc_eval(
            src,
            q,
            src,
            resolved_kmax,
            eps;
            compute_pot = (pg == 1),
            compute_grad = (pg == 2),
            nthreads = nthreads,
        )
    end

    if pgt > 0
        isnothing(targets) && throw(ArgumentError("targets are required when pgt > 0"))
        trg = as_3xn(targets)
        pottarg, gradtarg = _ltkm3dc_eval(
            src,
            q,
            trg,
            resolved_kmax,
            eps;
            compute_pot = (pgt == 1),
            compute_grad = (pgt == 2),
            nthreads = nthreads,
        )
    elseif !isnothing(targets)
        as_3xn(targets)
    end

    return TKMVals(pot, grad, pottarg, gradtarg, 0)
end
