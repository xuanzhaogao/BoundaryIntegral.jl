# sources contribute to the right hand side of the problem
abstract type AbstractSource end

# simple Point Source
struct PointSource{T, D} <: AbstractSource
    point::NTuple{D, T}
    charge::T
end

Base.show(io::IO, s::PointSource{T, D}) where {T, D} = print(io, "PointSource at $(s.point) with charge $(s.charge) in $T")

# may extend to other types of sources later, not sure about the structure

function rhs_approx(interface::DielectricInterface{FlatPanel{T, 3}, T}, ps::PointSource{T, 3}, eps_src::T; tol::T = sqrt(eps(T))) where T
    rhs(p, n) = -ps.charge * laplace3d_grad(ps.point, p, n) / eps_src
    return rhs_approx(interface, rhs; tol = tol)
end


struct VolumeSource{T, D} <: AbstractSource
    axes::NTuple{D, Vector{T}}
    weights::Array{T, D}
    density::Array{T, D}
    origin::NTuple{D, T}
    basis::NTuple{D, NTuple{D, T}}
end

struct VolumeSourcePointInfo{T}
    point::NTuple{3, T}
    weight::T
    density::T
    idx::NTuple{3, Int}
    global_idx::Int
end

struct VolumeSourceIterator{T}
    vs::VolumeSource{T, 3}
end

eachpoint(vs::VolumeSource{T, 3}) where {T} = VolumeSourceIterator{T}(vs)

Base.length(it::VolumeSourceIterator{T}) where {T} = begin
    xs, ys, zs = it.vs.axes
    return length(xs) * length(ys) * length(zs)
end

Base.eltype(::VolumeSourceIterator{T}) where {T} = VolumeSourcePointInfo{T}

function Base.iterate(it::VolumeSourceIterator{T}, state::NTuple{4, Int} = (1, 1, 1, 1)) where {T}
    ix, iy, iz, global_idx = state
    vs = it.vs
    xs, ys, zs = vs.axes
    nx = length(xs)
    ny = length(ys)
    nz = length(zs)

    ix > nx && return nothing

    point = volume_source_point(vs, ix, iy, iz)
    info = VolumeSourcePointInfo{T}(
        point,
        vs.weights[ix, iy, iz],
        vs.density[ix, iy, iz],
        (ix, iy, iz),
        global_idx,
    )

    if iz < nz
        next_state = (ix, iy, iz + 1, global_idx + 1)
    elseif iy < ny
        next_state = (ix, iy + 1, 1, global_idx + 1)
    else
        next_state = (ix + 1, 1, 1, global_idx + 1)
    end

    return (info, next_state)
end

function _identity_basis(::Type{T}, ::Val{3}) where {T}
    return (
        (one(T), zero(T), zero(T)),
        (zero(T), one(T), zero(T)),
        (zero(T), zero(T), one(T)),
    )
end

function _validate_volume_source_grid(
    axes::NTuple{3, Vector{T}},
    weights::Array{T, 3},
    density::Array{T, 3},
) where {T}
    nx, ny, nz = length(axes[1]), length(axes[2]), length(axes[3])
    size(weights) == (nx, ny, nz) || throw(ArgumentError("weights shape must match axes lengths"))
    size(density) == (nx, ny, nz) || throw(ArgumentError("density shape must match axes lengths"))
    return nothing
end

function VolumeSource{T, 3}(axes::NTuple{3, Vector{T}}, weights::Array{T, 3}, density::Array{T, 3}) where {T}
    _validate_volume_source_grid(axes, weights, density)
    return VolumeSource{T, 3}(axes, weights, density, (zero(T), zero(T), zero(T)), _identity_basis(T, Val(3)))
end

function VolumeSource(axes::NTuple{3, Vector{T}}, weights::Array{T, 3}, density::Array{T, 3}) where {T}
    return VolumeSource{T, 3}(axes, weights, density)
end

function VolumeSource(
    axes::NTuple{3, Vector{T}},
    weights::Array{T, 3},
    density::Array{T, 3},
    origin::NTuple{3, T},
    basis::NTuple{3, NTuple{3, T}},
) where {T}
    _validate_volume_source_grid(axes, weights, density)
    return VolumeSource{T, 3}(axes, weights, density, origin, basis)
end

function volume_source_point(vs::VolumeSource{T, 3}, ix::Int, iy::Int, iz::Int) where {T}
    u = vs.axes[1][ix]
    v = vs.axes[2][iy]
    w = vs.axes[3][iz]
    o = vs.origin
    a, b, c = vs.basis
    return (
        o[1] + u * a[1] + v * b[1] + w * c[1],
        o[2] + u * a[2] + v * b[2] + w * c[2],
        o[3] + u * a[3] + v * b[3] + w * c[3],
    )
end

function _is_uniform_axis(axis::AbstractVector{T}; rtol::T = sqrt(eps(T)), atol::T = zero(T)) where {T}
    n = length(axis)
    n <= 2 && return true
    step = axis[2] - axis[1]
    for i in 3:n
        if !isapprox(axis[i] - axis[i - 1], step; rtol = rtol, atol = atol)
            return false
        end
    end
    return true
end

function VolumeSource(points::Vector{NTuple{3, T}}, weights::Vector{T}, density::Vector{T}) where {T}
    n = length(points)
    n == length(weights) || throw(ArgumentError("points and weights must have the same length"))
    n == length(density) || throw(ArgumentError("points and density must have the same length"))

    xs = sort(unique(p[1] for p in points))
    ys = sort(unique(p[2] for p in points))
    zs = sort(unique(p[3] for p in points))
    nx, ny, nz = length(xs), length(ys), length(zs)
    n == nx * ny * nz || throw(ArgumentError("points must form a full tensor grid"))

    wx = Array{T, 3}(undef, nx, ny, nz)
    dens = Array{T, 3}(undef, nx, ny, nz)
    filled = falses(nx, ny, nz)

    for i in 1:n
        x, y, z = points[i]
        ix = searchsortedfirst(xs, x)
        iy = searchsortedfirst(ys, y)
        iz = searchsortedfirst(zs, z)
        (1 <= ix <= nx && 1 <= iy <= ny && 1 <= iz <= nz) || throw(ArgumentError("point out of grid bounds"))
        filled[ix, iy, iz] && throw(ArgumentError("duplicate point in grid"))
        wx[ix, iy, iz] = weights[i]
        dens[ix, iy, iz] = density[i]
        filled[ix, iy, iz] = true
    end

    all(filled) || throw(ArgumentError("points must cover the full tensor grid"))
    return VolumeSource((xs, ys, zs), wx, dens)
end

function GaussianVolumeSource(center::NTuple{3, T}, σ::T, n::Int, tol::T) where T
    @assert n >= 1 "n must be >= 1"
    @assert σ > zero(T) "σ must be > 0"
    @assert tol > zero(T) && tol < one(T) "tol must be in (0, 1)"

    ns, ws = gausslegendre(n)
    two_sigma2 = T(2) * σ * σ
    support_r = sqrt(two_sigma2 * log(inv(tol)))
    norm_factor = inv((sqrt(T(2) * T(pi)) * σ)^3)
    xs = [center[1] + support_r * T(ns[i]) for i in 1:n]
    ys = [center[2] + support_r * T(ns[i]) for i in 1:n]
    zs = [center[3] + support_r * T(ns[i]) for i in 1:n]
    weights = Array{T, 3}(undef, n, n, n)
    density = Array{T, 3}(undef, n, n, n)

    for i in 1:n
        xi = xs[i]
        wi = support_r * T(ws[i])
        for j in 1:n
            yj = ys[j]
            wj = support_r * T(ws[j])
            for k in 1:n
                zk = zs[k]
                wk = support_r * T(ws[k])
                weights[i, j, k] = wi * wj * wk
                r2 = (xi - center[1])^2 + (yj - center[2])^2 + (zk - center[3])^2
                density[i, j, k] = norm_factor * exp(-r2 / two_sigma2)
            end
        end
    end

    return VolumeSource((xs, ys, zs), weights, density)
end

function _gaussian_quad_order(σ::T, tol::T; n_max::Int = 64) where T
    @assert σ > zero(T) "σ must be > 0"
    @assert tol > zero(T) && tol < one(T) "tol must be in (0, 1)"
    @assert n_max >= 1 "n_max must be >= 1"

    two_sigma2 = T(2) * σ * σ
    support_r = sqrt(two_sigma2 * log(inv(tol)))
    norm_factor = inv((sqrt(T(2) * T(pi)) * σ)^3)

    n_sample = 10
    for n in 2:n_max
        ns, ws = gausslegendre(n)
        nsT = T.(ns)
        wsT = T.(ws)
        λ = gl_barycentric_weights(nsT, wsT)
        gx = [exp(-((support_r * nsT[i])^2) / two_sigma2) for i in 1:n]
        gy = gx
        gz = gx
        us = collect(LinRange(T(-1), T(1), n_sample))

        max_err = zero(T)
        for i in eachindex(us), j in eachindex(us), k in eachindex(us)
            rx = barycentric_row(nsT, λ, us[i])
            ry = barycentric_row(nsT, λ, us[j])
            rz = barycentric_row(nsT, λ, us[k])
            gx_i = sum(rx .* gx)
            gy_j = sum(ry .* gy)
            gz_k = sum(rz .* gz)
            approx = norm_factor * gx_i * gy_j * gz_k
            x = support_r * us[i]
            y = support_r * us[j]
            z = support_r * us[k]
            r2 = x^2 + y^2 + z^2
            exact = norm_factor * exp(-r2 / two_sigma2)
            err = abs(approx - exact)
            max_err = max(max_err, err)
        end

        if max_err <= tol
            return n
        end
    end

    throw(ArgumentError("could not resolve Gaussian with tol=$(tol) using n <= $(n_max)"))
end

function GaussianVolumeSource(center::NTuple{3, T}, σ::T, tol::T) where T
    n = _gaussian_quad_order(σ, tol)
    return GaussianVolumeSource(center, σ, n, tol)
end
