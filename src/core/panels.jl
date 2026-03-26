abstract type AbstractPanel end

# flat panel in D-dimensional space
# discretized by tensor product gauss-legendre quadrature
struct FlatPanel{T, D} <: AbstractPanel
    # information about the panel
    normal::NTuple{D, T}
    corners::Vector{NTuple{D, T}} # corners are arranged in a anti-clockwise order (lb, rb, rt, lt)
    is_edge::Bool

    # quadrature information (same order in each tangential direction)
    n_quad::Int
    gl_xs::Vector{T}
    gl_ws::Vector{T}

    # quadrature points and weights
    points::Vector{NTuple{D, T}}
    weights::Vector{T}

    # barycentric interpolation weights on GL nodes
    bary_weights::Vector{T}
end

# Convenience constructor: computes barycentric weights automatically
function FlatPanel(normal::NTuple{D,T}, corners, is_edge, n_quad, gl_xs, gl_ws, points, weights) where {T, D}
    bary_weights = gl_barycentric_weights(gl_xs, gl_ws)
    return FlatPanel{T,D}(normal, corners, is_edge, n_quad, gl_xs, gl_ws, points, weights, bary_weights)
end

Base.show(io::IO, p::FlatPanel{T, D}) where {T, D} = print(io, "FlatPanel in $D-dimensional space, with $(p.n_quad) quadrature points in $T")

num_points(p::FlatPanel) = length(p.points)

# Iterator for FlatPanel - iterates over quadrature points
struct FlatPanelPointInfo{T, D}
    point::NTuple{D, T}
    normal::NTuple{D, T}
    weight::T
    idx::Int
end

struct FlatPanelPointIterator{T, D}
    panel::FlatPanel{T, D}
end

eachpoint(p::FlatPanel) = FlatPanelPointIterator(p)

Base.length(it::FlatPanelPointIterator) = length(it.panel.points)
Base.eltype(::FlatPanelPointIterator{T, D}) where {T, D} = FlatPanelPointInfo{T, D}

function Base.iterate(it::FlatPanelPointIterator{T, D}, idx::Int = 1) where {T, D}
    idx > length(it.panel.points) && return nothing
    info = FlatPanelPointInfo{T, D}(
        it.panel.points[idx],
        it.panel.normal,
        it.panel.weights[idx],
        idx
    )
    return (info, idx + 1)
end

# dielectric interfaces is a collection of panels with different dielectric constants inside and outsize
# in and out is defined accroding to normal direction of the panel
struct DielectricInterface{P <: AbstractPanel, T}
    panels::Vector{P}
    eps_in::Vector{T}
    eps_out::Vector{T}
end

Base.show(io::IO, d::DielectricInterface{P, T}) where {P <: AbstractPanel, T} = print(io, "DielectricInterface with $(length(d.panels)) panels in $T")

# Iterator for DielectricInterface - iterates over all quadrature points across all panels
struct DielectricPointInfo{T, D}
    panel_point::FlatPanelPointInfo{T, D}
    eps_in::T
    eps_out::T
    global_idx::Int
    panel_idx::Int
end

struct DielectricInterfaceIterator{P <: AbstractPanel, T}
    d::DielectricInterface{P, T}
end

eachpoint(d::DielectricInterface) = DielectricInterfaceIterator(d)

function Base.iterate(it::DielectricInterfaceIterator{P, T}, state = (1, nothing, 1)) where {P <: AbstractPanel, T}
    panel_idx, panel_iter_state, global_idx = state
    
    # Check if we've exhausted all panels
    panel_idx > length(it.d.panels) && return nothing
    
    panel = it.d.panels[panel_idx]
    panel_iter = eachpoint(panel)
    
    # Get next point from current panel's iterator
    result = panel_iter_state === nothing ? iterate(panel_iter) : iterate(panel_iter, panel_iter_state)
    
    # Move to next panel if current panel is exhausted
    if result === nothing
        return Base.iterate(it, (panel_idx + 1, nothing, global_idx))
    end
    
    panel_point, next_panel_state = result
    D = length(panel.normal)
    info = DielectricPointInfo{T, D}(
        panel_point,
        it.d.eps_in[panel_idx],
        it.d.eps_out[panel_idx],
        global_idx,
        panel_idx
    )
    
    return (info, (panel_idx, next_panel_state, global_idx + 1))
end

num_points(d::DielectricInterface) = sum(num_points(panel) for panel in d.panels)
all_weights(d::DielectricInterface) = vcat([panel.weights for panel in d.panels]...)

Base.length(it::DielectricInterfaceIterator) = num_points(it.d)

struct PanelRhsInterp{T}
    cc::NTuple{3, T}
    bma::NTuple{3, T}
    dma::NTuple{3, T}
    inv11::T
    inv12::T
    inv22::T
    scale::T
    n_quad::Int
    ns::Vector{T}
    bary_weights::Vector{T}
    vals::Matrix{T}
end

function _panel_uv(interp::PanelRhsInterp{T}, p::NTuple{3, T}, tol::T) where T
    x = p .- interp.cc
    rhs1 = dot(interp.bma, x)
    rhs2 = dot(interp.dma, x)
    s = interp.inv11 * rhs1 + interp.inv12 * rhs2
    t = interp.inv12 * rhs1 + interp.inv22 * rhs2
    u = 2 * s
    v = 2 * t
    residual = x .- interp.bma .* s .- interp.dma .* t
    on_plane = norm(residual) <= tol * interp.scale
    in_panel = abs(u) <= 1 + tol && abs(v) <= 1 + tol
    return u, v, on_plane && in_panel
end

function interface_approx(interface::DielectricInterface{FlatPanel{T, 3}, T}, values::AbstractVector{T}; tol::T = sqrt(eps(T))) where T
    @assert length(values) == num_points(interface)
    n_panels = length(interface.panels)
    interps = Vector{PanelRhsInterp{T}}(undef, n_panels)
    centers = Matrix{T}(undef, 3, n_panels)
    offset = 0
    for (i, panel) in enumerate(interface.panels)
        a, b, c, d = panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a
        dma = d .- a
        bb = dot(bma, bma)
        bd = dot(bma, dma)
        dd = dot(dma, dma)
        det = bb * dd - bd * bd
        inv11 = dd / det
        inv12 = -bd / det
        inv22 = bb / det
        scale = max(norm(bma), norm(dma))

        n_quad = panel.n_quad
        vals = Matrix{T}(undef, n_quad, n_quad)
        idx = offset + 1
        for ii in 1:n_quad
            for jj in 1:n_quad
                vals[ii, jj] = values[idx]
                idx += 1
            end
        end
        offset += n_quad * n_quad

        interps[i] = PanelRhsInterp(cc, bma, dma, inv11, inv12, inv22, scale, n_quad, panel.gl_xs, panel.bary_weights, vals)
        centers[1, i] = cc[1]
        centers[2, i] = cc[2]
        centers[3, i] = cc[3]
    end

    tree = KDTree(centers)
    k = min(n_panels, 16)

    function approx(p::NTuple{3, T})
        idxs, _ = knn(tree, collect(p), k, true)
        for idx in idxs
            interp = interps[idx]
            u, v, hit = _panel_uv(interp, p, tol)
            if hit
                nq = interp.n_quad
                rx = Vector{T}(undef, nq)
                ry = Vector{T}(undef, nq)
                barycentric_row!(rx, interp.ns, interp.bary_weights, u)
                barycentric_row!(ry, interp.ns, interp.bary_weights, v)
                val = zero(T)
                for ii in 1:nq
                    for jj in 1:nq
                        val += interp.vals[ii, jj] * rx[ii] * ry[jj]
                    end
                end
                return val
            end
        end
        for interp in interps
            u, v, hit = _panel_uv(interp, p, tol)
            if hit
                nq = interp.n_quad
                rx = Vector{T}(undef, nq)
                ry = Vector{T}(undef, nq)
                barycentric_row!(rx, interp.ns, interp.bary_weights, u)
                barycentric_row!(ry, interp.ns, interp.bary_weights, v)
                val = zero(T)
                for ii in 1:nq
                    for jj in 1:nq
                        val += interp.vals[ii, jj] * rx[ii] * ry[jj]
                    end
                end
                return val
            end
        end
        error("point is not on any panel within tolerance")
    end

    return approx
end

function rhs_approx(interface::DielectricInterface{FlatPanel{T, 3}, T}, rhs::Function; tol::T = sqrt(eps(T))) where T
    values = Vector{T}(undef, num_points(interface))
    for (i, point) in enumerate(eachpoint(interface))
        values[i] = T(rhs(point.panel_point.point, point.panel_point.normal))
    end
    return interface_approx(interface, values; tol = tol)
end

function interface_uniform_samples(
    interface::DielectricInterface{FlatPanel{T, 3}, T},
    values::AbstractVector{<:Real};
    n_sample::Int = 20,
    tol::T = sqrt(eps(T)),
) where T
    n_sample >= 2 || throw(ArgumentError("n_sample must be >= 2"))
    length(values) == num_points(interface) || throw(ArgumentError("values length must match num_points(interface)"))

    vals = T.(values)
    approx = interface_approx(interface, vals; tol = tol)
    # Group panels by coplanar surface (normal + plane offset).
    function canonical_normal(n::NTuple{3, T})
        nn = norm(n)
        nn > zero(T) || throw(ArgumentError("panel normal has zero norm"))
        nu = n ./ nn
        sign_flip = (nu[1] < -tol) || (abs(nu[1]) <= tol && nu[2] < -tol) || (abs(nu[1]) <= tol && abs(nu[2]) <= tol && nu[3] < -tol)
        return sign_flip ? (-nu[1], -nu[2], -nu[3]) : nu
    end

    quant_scale = T(1e6)
    q(x) = round(Int, x * quant_scale)
    surface_groups = Dict{NTuple{4, Int}, Vector{Int}}()

    for (pi, panel) in enumerate(interface.panels)
        n = canonical_normal(panel.normal)
        a, b, c, d = panel.corners
        center = (a .+ b .+ c .+ d) ./ 4
        offset = dot(n, center)
        key = (q(n[1]), q(n[2]), q(n[3]), q(offset))
        push!(get!(surface_groups, key, Int[]), pi)
    end

    function cross3(a::NTuple{3, T}, b::NTuple{3, T})
        return (
            a[2] * b[3] - a[3] * b[2],
            a[3] * b[1] - a[1] * b[3],
            a[1] * b[2] - a[2] * b[1],
        )
    end

    samples = Vector{NamedTuple{(:X, :Y, :Z, :V), Tuple{Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}}}}()

    for panel_ids in Base.values(surface_groups)
        first_panel = interface.panels[first(panel_ids)]
        n = canonical_normal(first_panel.normal)

        a0, b0, c0, d0 = first_panel.corners
        edge = b0 .- a0
        edge_n = norm(edge)
        if edge_n <= tol
            edge = d0 .- a0
            edge_n = norm(edge)
        end
        edge_n > tol || throw(ArgumentError("could not infer tangential axis for surface"))
        e1 = edge ./ edge_n
        e2_raw = cross3(n, e1)
        e2_n = norm(e2_raw)
        e2_n > tol || throw(ArgumentError("could not infer second tangential axis for surface"))
        e2 = e2_raw ./ e2_n

        origin = a0
        smin = typemax(T)
        smax = typemin(T)
        tmin = typemax(T)
        tmax = typemin(T)
        for pid in panel_ids
            panel = interface.panels[pid]
            for corner in panel.corners
                rel = corner .- origin
                s = dot(rel, e1)
                t = dot(rel, e2)
                smin = min(smin, s)
                smax = max(smax, s)
                tmin = min(tmin, t)
                tmax = max(tmax, t)
            end
        end

        ss = collect(LinRange(smin, smax, n_sample))
        tt = collect(LinRange(tmin, tmax, n_sample))
        X = Matrix{T}(undef, n_sample, n_sample)
        Y = Matrix{T}(undef, n_sample, n_sample)
        Z = Matrix{T}(undef, n_sample, n_sample)
        V = Matrix{T}(undef, n_sample, n_sample)

        for j in 1:n_sample
            t = tt[j]
            for i in 1:n_sample
                s = ss[i]
                p = origin .+ e1 .* s .+ e2 .* t
                X[i, j] = p[1]
                Y[i, j] = p[2]
                Z[i, j] = p[3]
                V[i, j] = approx((p[1], p[2], p[3]))
            end
        end

        push!(samples, (X = X, Y = Y, Z = Z, V = V))
    end

    return samples
end
