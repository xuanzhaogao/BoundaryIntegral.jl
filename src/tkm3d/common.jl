struct TKMVals{P, G, PT, GT}
    pot::P
    grad::G
    pottarg::PT
    gradtarg::GT
    ier::Int
end

struct KCut3DCResult{T}
    kcut::T
    kmax_nyquist::T
    axis_nyquist::NTuple{3, T}
    max_coeff::T
    tail_ratio::T
    nmodes::NTuple{3, Int}
    Δk::NTuple{3, T}
end

function as_3xn(points::AbstractMatrix{<:Real})
    size(points, 1) == 3 || throw(ArgumentError("points must have shape (3, N)"))
    return Matrix{float(eltype(points))}(points)
end

function combined_box_geometry_3xn(sources::AbstractMatrix{T}, targets::AbstractMatrix{T}) where {T <: Real}
    mins = vec(min.(minimum(sources; dims = 2), minimum(targets; dims = 2)))
    maxs = vec(max.(maximum(sources; dims = 2), maximum(targets; dims = 2)))
    lengths = maxs .- mins
    center = (mins .+ maxs) ./ 2
    return lengths, center
end

function centered_mode_axis(Δk::T, k_max::T) where {T <: Real}
    Δk > zero(T) || throw(ArgumentError("mode spacing must be positive"))
    k_max >= zero(T) || throw(ArgumentError("k_max must be nonnegative"))

    mmax = ceil(Int, k_max / Δk)
    nmodes = 2 * mmax + 1
    axis = Vector{T}(undef, nmodes)
    shift = nmodes ÷ 2
    for i in 1:nmodes
        axis[i] = Δk * (i - 1 - shift)
    end
    return axis
end

function _pointwise_tail_ratio(
    radii::AbstractVector{<:Real},
    magnitudes::AbstractVector{<:Real},
    cutoff::Real,
)
    length(radii) == length(magnitudes) || throw(ArgumentError("radii and magnitudes must have the same length"))
    isempty(radii) && throw(ArgumentError("spectrum cannot be empty"))

    max_mag = maximum(magnitudes)
    max_mag > 0 || return 0.0

    tail_max = 0.0
    @inbounds for i in eachindex(radii)
        if radii[i] > cutoff && magnitudes[i] > tail_max
            tail_max = Float64(magnitudes[i])
        end
    end
    return tail_max / max_mag
end

function _smallest_kcut_from_tail(
    radii::AbstractVector{<:Real},
    magnitudes::AbstractVector{<:Real},
    tol::Real,
)
    length(radii) == length(magnitudes) || throw(ArgumentError("radii and magnitudes must have the same length"))
    isempty(radii) && throw(ArgumentError("spectrum cannot be empty"))
    0.0 < tol < 1.0 || throw(ArgumentError("tol must satisfy 0 < tol < 1"))

    max_mag = maximum(magnitudes)
    max_mag > 0 || return 0.0

    order = sortperm(radii)
    sorted_radii = Float64.(radii[order])
    sorted_magnitudes = Float64.(magnitudes[order])
    suffix_exclusive = Vector{Float64}(undef, length(sorted_magnitudes))
    suffix_exclusive[end] = 0.0

    @inbounds for i in (length(sorted_magnitudes) - 1):-1:1
        suffix_exclusive[i] = max(suffix_exclusive[i + 1], sorted_magnitudes[i + 1])
    end

    i = 1
    while i <= length(sorted_radii)
        j = i
        while j < length(sorted_radii) && sorted_radii[j + 1] == sorted_radii[i]
            j += 1
        end
        if suffix_exclusive[i] / max_mag < tol
            return sorted_radii[i]
        end
        i = j + 1
    end

    return sorted_radii[end]
end

function _spectral_radii_and_magnitudes(
    coeff::AbstractArray{<:Complex, 3},
    kx::AbstractVector{<:Real},
    ky::AbstractVector{<:Real},
    kz::AbstractVector{<:Real},
)
    size(coeff) == (length(kx), length(ky), length(kz)) ||
        throw(ArgumentError("coefficient array shape must match mode axes"))

    npts = length(coeff)
    radii = Vector{Float64}(undef, npts)
    magnitudes = Vector{Float64}(undef, npts)

    idx = 1
    @inbounds for iz in eachindex(kz), iy in eachindex(ky), ix in eachindex(kx)
        radii[idx] = sqrt(kx[ix]^2 + ky[iy]^2 + kz[iz]^2)
        magnitudes[idx] = abs(coeff[ix, iy, iz])
        idx += 1
    end

    return radii, magnitudes
end

function _finufft_make_type2_plan_3d(
    xj::AbstractVector{T},
    yj::AbstractVector{T},
    zj::AbstractVector{T},
    iflag::Integer,
    eps::Real,
    nmodes::NTuple{3, Int},
    ntrans::Integer,
    dtype::Type{T};
    kwargs...,
) where {T <: AbstractFloat}
    modes = FINUFFT.BIGINT[nmodes...]
    plan = FINUFFT.finufft_makeplan(2, modes, iflag, ntrans, eps; dtype = dtype, kwargs...)
    FINUFFT.finufft_setpts!(plan, collect(T, xj), collect(T, yj), collect(T, zj))
    return plan
end

function _finufft_type2_exec_3d(plan, fk::Array{Complex{T}, 3}) where {T <: AbstractFloat}
    return vec(FINUFFT.finufft_exec(plan, fk))
end

function _finufft_type2_exec_3d(plan, fk::Array{Complex{T}, 4}) where {T <: AbstractFloat}
    return FINUFFT.finufft_exec(plan, fk)
end

function _finufft_type2_eval_3d(
    xj::AbstractVector{T},
    yj::AbstractVector{T},
    zj::AbstractVector{T},
    iflag::Integer,
    eps::Real,
    fk::Array{Complex{T}, N};
    kwargs...,
) where {T <: AbstractFloat, N}
    N in (3, 4) || throw(ArgumentError("fk must be a 3D or 4D complex array"))
    nmodes = size(fk)[1:3]
    ntrans = N == 3 ? 1 : size(fk, 4)
    plan = _finufft_make_type2_plan_3d(xj, yj, zj, iflag, eps, nmodes, ntrans, T; kwargs...)
    try
        return _finufft_type2_exec_3d(plan, fk)
    finally
        FINUFFT.finufft_destroy!(plan)
    end
end

function _spectral_gradient_coeffs_3d(
    coeff::Array{Complex{T}, 3},
    kx::AbstractVector{T},
    ky::AbstractVector{T},
    kz::AbstractVector{T},
) where {T <: AbstractFloat}
    grad_coeff = Array{Complex{T}, 4}(undef, size(coeff)..., 3)
    @inbounds for iz in eachindex(kz), iy in eachindex(ky), ix in eachindex(kx)
        c = coeff[ix, iy, iz]
        grad_coeff[ix, iy, iz, 1] = (im * kx[ix]) * c
        grad_coeff[ix, iy, iz, 2] = (im * ky[iy]) * c
        grad_coeff[ix, iy, iz, 3] = (im * kz[iz]) * c
    end
    return grad_coeff
end

@inline function truncated_laplace3d_hat(k::T, L::T) where {T <: Real}
    if iszero(k)
        return L^2 / 2
    end
    return 2 * (sin(L * k / 2) / k)^2
end
