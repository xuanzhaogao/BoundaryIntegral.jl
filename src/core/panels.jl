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

    # quadrature points and weightsnorm(lhs * x - rhs)
    points::Vector{NTuple{D, T}}
    weights::Vector{T}
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
