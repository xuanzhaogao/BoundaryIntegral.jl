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
    positions::Matrix{T}
    weights::Vector{T}
    density::Vector{T}
end

struct VolumeSourcePointInfo{T}
    point::NTuple{3, T}
    weight::T
    density::T
    idx::Int
    global_idx::Int
end

struct VolumeSourceIterator{T}
    vs::VolumeSource{T, 3}
end

eachpoint(vs::VolumeSource{T, 3}) where {T} = VolumeSourceIterator{T}(vs)

Base.length(it::VolumeSourceIterator{T}) where {T} = length(it.vs.density)

Base.eltype(::VolumeSourceIterator{T}) where {T} = VolumeSourcePointInfo{T}

function Base.iterate(it::VolumeSourceIterator{T}, state::Int = 1) where {T}
    idx = state
    vs = it.vs
    idx > length(vs.density) && return nothing

    point = volume_source_point(vs, idx)
    info = VolumeSourcePointInfo{T}(
        point,
        vs.weights[idx],
        vs.density[idx],
        idx,
        idx,
    )
    next_state = idx + 1
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

function _validate_volume_source_flat(
    positions::AbstractMatrix{T},
    weights::AbstractVector{T},
    density::AbstractVector{T},
) where {T}
    size(positions, 1) == 3 || throw(ArgumentError("positions must have shape (3, n)"))
    n = size(positions, 2)
    n == length(weights) || throw(ArgumentError("positions and weights must have the same length"))
    n == length(density) || throw(ArgumentError("positions and density must have the same length"))
    return n
end

function _truncate_volume_source(
    positions::Matrix{T},
    weights::Vector{T},
    density::Vector{T},
    tol::T,
) where {T}
    tol < zero(T) && throw(ArgumentError("tol must be >= 0"))
    tol == zero(T) && return positions, weights, density

    keep = abs.(density) .>= tol
    any(keep) || throw(ArgumentError("all source points are truncated by tol=$(tol)"))
    return positions[:, keep], weights[keep], density[keep]
end

function VolumeSource(
    positions::AbstractMatrix{T},
    weights::AbstractVector{T},
    density::AbstractVector{T};
    tol::Real = 0,
) where {T}
    _validate_volume_source_flat(positions, weights, density)
    pos = Matrix{T}(positions)
    w = Vector{T}(weights)
    rho = Vector{T}(density)
    pos_t, w_t, rho_t = _truncate_volume_source(pos, w, rho, T(tol))
    return VolumeSource{T, 3}(pos_t, w_t, rho_t)
end

function _volume_source_positions(
    axes::NTuple{3, Vector{T}},
    origin::NTuple{3, T},
    basis::NTuple{3, NTuple{3, T}},
) where {T}
    xs, ys, zs = axes
    nx, ny, nz = length(xs), length(ys), length(zs)
    n = nx * ny * nz
    positions = Matrix{T}(undef, 3, n)
    idx = 0
    o = origin
    a, b, c = basis
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        idx += 1
        u = xs[ix]
        v = ys[iy]
        w = zs[iz]
        positions[1, idx] = o[1] + u * a[1] + v * b[1] + w * c[1]
        positions[2, idx] = o[2] + u * a[2] + v * b[2] + w * c[2]
        positions[3, idx] = o[3] + u * a[3] + v * b[3] + w * c[3]
    end
    return positions
end

function _flatten_volume_source_data(weights::Array{T, 3}, density::Array{T, 3}) where {T}
    nx, ny, nz = size(weights)
    w = Vector{T}(undef, nx * ny * nz)
    rho = Vector{T}(undef, nx * ny * nz)
    idx = 0
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        idx += 1
        w[idx] = weights[ix, iy, iz]
        rho[idx] = density[ix, iy, iz]
    end
    return w, rho
end

function VolumeSource{T, 3}(axes::NTuple{3, Vector{T}}, weights::Array{T, 3}, density::Array{T, 3}; tol::Real = 0) where {T}
    _validate_volume_source_grid(axes, weights, density)
    origin = (zero(T), zero(T), zero(T))
    basis = _identity_basis(T, Val(3))
    positions = _volume_source_positions(axes, origin, basis)
    w, rho = _flatten_volume_source_data(weights, density)
    return VolumeSource(positions, w, rho; tol = tol)
end

function VolumeSource(axes::NTuple{3, Vector{T}}, weights::Array{T, 3}, density::Array{T, 3}; tol::Real = 0) where {T}
    return VolumeSource{T, 3}(axes, weights, density; tol = tol)
end

function VolumeSource(
    axes::NTuple{3, Vector{T}},
    weights::Array{T, 3},
    density::Array{T, 3},
    origin::NTuple{3, T},
    basis::NTuple{3, NTuple{3, T}};
    tol::Real = 0,
) where {T}
    _validate_volume_source_grid(axes, weights, density)
    positions = _volume_source_positions(axes, origin, basis)
    w, rho = _flatten_volume_source_data(weights, density)
    return VolumeSource(positions, w, rho; tol = tol)
end

function volume_source_point(vs::VolumeSource{T, 3}, idx::Int) where {T}
    1 <= idx <= size(vs.positions, 2) || throw(BoundsError(vs.positions, (Colon(), idx)))
    return (
        vs.positions[1, idx],
        vs.positions[2, idx],
        vs.positions[3, idx],
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

function VolumeSource(points::Vector{NTuple{3, T}}, weights::Vector{T}, density::Vector{T}; tol::Real = 0) where {T}
    n = length(points)
    n == length(weights) || throw(ArgumentError("points and weights must have the same length"))
    n == length(density) || throw(ArgumentError("points and density must have the same length"))
    positions = Matrix{T}(undef, 3, n)
    for i in 1:n
        x, y, z = points[i]
        positions[1, i] = x
        positions[2, i] = y
        positions[3, i] = z
    end
    return VolumeSource(positions, weights, density; tol = tol)
end

abstract type AbstractScreeningMode end

struct SharpScreening <: AbstractScreeningMode end

struct SoftMixPermittivity{T} <: AbstractScreeningMode
    bandwidth::T
    function SoftMixPermittivity(bandwidth::T) where {T}
        bandwidth > zero(T) || throw(ArgumentError("bandwidth must be positive"))
        return new{T}(bandwidth)
    end
end

struct SoftMixInversePermittivity{T} <: AbstractScreeningMode
    bandwidth::T
    function SoftMixInversePermittivity(bandwidth::T) where {T}
        bandwidth > zero(T) || throw(ArgumentError("bandwidth must be positive"))
        return new{T}(bandwidth)
    end
end

function _interface_box_bounds(interface::DielectricInterface{P, T}) where {T, P <: FlatPanel{T, 3}}
    isempty(interface.panels) && throw(ArgumentError("interface must contain at least one panel"))

    min_corner = (typemax(T), typemax(T), typemax(T))
    max_corner = (typemin(T), typemin(T), typemin(T))
    for panel in interface.panels
        for corner in panel.corners
            min_corner = (
                min(min_corner[1], corner[1]),
                min(min_corner[2], corner[2]),
                min(min_corner[3], corner[3]),
            )
            max_corner = (
                max(max_corner[1], corner[1]),
                max(max_corner[2], corner[2]),
                max(max_corner[3], corner[3]),
            )
        end
    end

    return min_corner, max_corner
end

function _uniform_interface_eps(interface::DielectricInterface{P, T}) where {T, P <: FlatPanel{T, 3}}
    isempty(interface.panels) && throw(ArgumentError("interface must contain at least one panel"))
    eps_in = interface.eps_in[1]
    eps_out = interface.eps_out[1]
    all(isequal(eps_in), interface.eps_in) || throw(ArgumentError("screened_volume_source requires a uniform eps_in across the interface"))
    all(isequal(eps_out), interface.eps_out) || throw(ArgumentError("screened_volume_source requires a uniform eps_out across the interface"))
    return eps_in, eps_out
end

@inline function _point_in_box(point::NTuple{3, T}, min_corner::NTuple{3, T}, max_corner::NTuple{3, T}, tol::T) where {T}
    return (min_corner[1] - tol <= point[1] <= max_corner[1] + tol) &&
        (min_corner[2] - tol <= point[2] <= max_corner[2] + tol) &&
        (min_corner[3] - tol <= point[3] <= max_corner[3] + tol)
end

@inline function _smooth_screen_factor(distance::T, bandwidth::T) where {T}
    return (one(T) + erf(distance / bandwidth)) / T(2)
end

@inline function _box_screen(point::NTuple{3, T}, min_corner::NTuple{3, T}, max_corner::NTuple{3, T}, bandwidth::T) where {T}
    return _smooth_screen_factor(point[1] - min_corner[1], bandwidth) *
        _smooth_screen_factor(max_corner[1] - point[1], bandwidth) *
        _smooth_screen_factor(point[2] - min_corner[2], bandwidth) *
        _smooth_screen_factor(max_corner[2] - point[2], bandwidth) *
        _smooth_screen_factor(point[3] - min_corner[3], bandwidth) *
        _smooth_screen_factor(max_corner[3] - point[3], bandwidth)
end

@inline function _screened_permittivity(::SharpScreening, point::NTuple{3, T}, min_corner::NTuple{3, T}, max_corner::NTuple{3, T}, eps_in::T, eps_out::T, tol::T) where {T}
    return _point_in_box(point, min_corner, max_corner, tol) ? eps_in : eps_out
end

@inline function _screened_permittivity(mode::SoftMixPermittivity, point::NTuple{3, T}, min_corner::NTuple{3, T}, max_corner::NTuple{3, T}, eps_in::T, eps_out::T, ::T) where {T}
    bandwidth = T(mode.bandwidth)
    s = _box_screen(point, min_corner, max_corner, bandwidth)
    return eps_in * s + eps_out * (one(T) - s)
end

@inline function _screened_permittivity(mode::SoftMixInversePermittivity, point::NTuple{3, T}, min_corner::NTuple{3, T}, max_corner::NTuple{3, T}, eps_in::T, eps_out::T, ::T) where {T}
    bandwidth = T(mode.bandwidth)
    s = _box_screen(point, min_corner, max_corner, bandwidth)
    return inv(s / eps_in + (one(T) - s) / eps_out)
end

function _screened_volume_density(
    vs::VolumeSource{T, 3},
    min_corner::NTuple{3, T},
    max_corner::NTuple{3, T},
    eps_in::T,
    eps_out::T,
    mode::AbstractScreeningMode,
    tol::T,
) where {T}
    rho = similar(vs.density)
    for s in eachindex(vs.density)
        pos = volume_source_point(vs, s)
        eps_local = _screened_permittivity(mode, pos, min_corner, max_corner, eps_in, eps_out, tol)
        rho[s] = vs.density[s] / eps_local
    end
    return rho
end

"""
    screened_volume_source(Lx, Ly, Lz, vs, eps_in, eps_out, mode; tol = ...)
    screened_volume_source(interface, vs, mode; tol = ...)

Apply the screened density selected by `mode` to each volume sample.
"""
function screened_volume_source(
    Lx::T,
    Ly::T,
    Lz::T,
    vs::VolumeSource{T, 3},
    eps_in::T,
    eps_out::T,
    mode::AbstractScreeningMode;
    tol::T = sqrt(eps(T)) * max(max(abs(Lx), abs(Ly)), abs(Lz)),
) where {T}
    min_corner = (-Lx / 2, -Ly / 2, -Lz / 2)
    max_corner = (Lx / 2, Ly / 2, Lz / 2)
    rho = _screened_volume_density(vs, min_corner, max_corner, eps_in, eps_out, mode, tol)
    return VolumeSource(copy(vs.positions), copy(vs.weights), rho)
end

function screened_volume_source(
    interface::DielectricInterface{P, T},
    vs::VolumeSource{T, 3},
    mode::AbstractScreeningMode;
    tol::T = begin
        min_corner, max_corner = _interface_box_bounds(interface)
        sqrt(eps(T)) * max(
            max(abs(max_corner[1] - min_corner[1]), abs(max_corner[2] - min_corner[2])),
            abs(max_corner[3] - min_corner[3]),
        )
    end,
) where {T, P <: FlatPanel{T, 3}}
    min_corner, max_corner = _interface_box_bounds(interface)
    eps_in, eps_out = _uniform_interface_eps(interface)
    rho = _screened_volume_density(vs, min_corner, max_corner, eps_in, eps_out, mode, tol)
    return VolumeSource(copy(vs.positions), copy(vs.weights), rho)
end

function GaussianVolumeSource(center::NTuple{3, T}, σ::T, n::Int, tol::T) where T
    @assert n >= 1 "n must be >= 1"
    @assert σ > zero(T) "σ must be > 0"
    @assert tol > zero(T) && tol < one(T) "tol must be in (0, 1)"

    two_sigma2 = T(2) * σ * σ
    support_r = sqrt(two_sigma2 * log(inv(tol)))
    norm_factor = inv((sqrt(T(2) * T(pi)) * σ)^3)
    h = T(2) * support_r / T(n)
    offset0 = -support_r + h / T(2)
    xs = [center[1] + offset0 + (i - 1) * h for i in 1:n]
    ys = [center[2] + offset0 + (i - 1) * h for i in 1:n]
    zs = [center[3] + offset0 + (i - 1) * h for i in 1:n]
    weights = Array{T, 3}(undef, n, n, n)
    density = Array{T, 3}(undef, n, n, n)

    for i in 1:n
        xi = xs[i]
        wi = h
        for j in 1:n
            yj = ys[j]
            wj = h
            for k in 1:n
                zk = zs[k]
                wk = h
                weights[i, j, k] = wi * wj * wk
                r2 = (xi - center[1])^2 + (yj - center[2])^2 + (zk - center[3])^2
                density[i, j, k] = norm_factor * exp(-r2 / two_sigma2)
            end
        end
    end

    return VolumeSource((xs, ys, zs), weights, density)
end
