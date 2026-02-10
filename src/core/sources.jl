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

function _resample_volume_to_uniform(axes::NTuple{3, Vector{T}}, density::AbstractArray{T, 3}) where {T}
    xs, ys, zs = axes
    nx, ny, nz = length(xs), length(ys), length(zs)

    xsu = collect(LinRange(first(xs), last(xs), nx))
    ysu = collect(LinRange(first(ys), last(ys), ny))
    zsu = collect(LinRange(first(zs), last(zs), nz))

    FT = float(T)
    out = Array{FT, 3}(undef, nx, ny, nz)

    for i in 1:nx
        x = xsu[i]
        ix = searchsortedlast(xs, x)
        ix = clamp(ix, 1, nx - 1)
        x0 = xs[ix]
        x1 = xs[ix + 1]
        tx = x1 == x0 ? zero(FT) : FT((x - x0) / (x1 - x0))
        for j in 1:ny
            y = ysu[j]
            iy = searchsortedlast(ys, y)
            iy = clamp(iy, 1, ny - 1)
            y0 = ys[iy]
            y1 = ys[iy + 1]
            ty = y1 == y0 ? zero(FT) : FT((y - y0) / (y1 - y0))
            for k in 1:nz
                z = zsu[k]
                iz = searchsortedlast(zs, z)
                iz = clamp(iz, 1, nz - 1)
                z0 = zs[iz]
                z1 = zs[iz + 1]
                tz = z1 == z0 ? zero(FT) : FT((z - z0) / (z1 - z0))

                v000 = FT(density[ix, iy, iz])
                v100 = FT(density[ix + 1, iy, iz])
                v010 = FT(density[ix, iy + 1, iz])
                v110 = FT(density[ix + 1, iy + 1, iz])
                v001 = FT(density[ix, iy, iz + 1])
                v101 = FT(density[ix + 1, iy, iz + 1])
                v011 = FT(density[ix, iy + 1, iz + 1])
                v111 = FT(density[ix + 1, iy + 1, iz + 1])

                c00 = v000 * (1 - tx) + v100 * tx
                c10 = v010 * (1 - tx) + v110 * tx
                c01 = v001 * (1 - tx) + v101 * tx
                c11 = v011 * (1 - tx) + v111 * tx
                c0 = c00 * (1 - ty) + c10 * ty
                c1 = c01 * (1 - ty) + c11 * ty
                out[i, j, k] = c0 * (1 - tz) + c1 * tz
            end
        end
    end

    return (xsu, ysu, zsu), out
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
    return VolumeSource{T, 3}((xs, ys, zs), wx, dens)
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

    return VolumeSource{T, 3}((xs, ys, zs), weights, density)
end
