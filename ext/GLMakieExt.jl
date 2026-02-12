module GLMakieExt

using GLMakie
using BoundaryIntegral
using BoundaryIntegral: VolumeSource
using BoundaryIntegral: AbstractPanel, DielectricInterface, build_neighbor_list

import BoundaryIntegral: viz_2d, viz_3d

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

function viz_2d!(ax::Axis, interface::DielectricInterface{P, T}; show_normals::Bool = true) where {P <: AbstractPanel, T}
    t = 0.05

    for panel in interface.panels
        a, b = panel.corners
        lines!(ax, [a[1], b[1]], [a[2], b[2]], color = :blue)

        show_normals || continue
        norm = panel.normal
        for (x1, x2) in panel.points
            lines!(ax, [x1, x1 + t * norm[1]], [x2, x2 + t * norm[2]], color = :black, linewidth = 0.2)
        end
    end
end

function viz_2d(interface::DielectricInterface{P, T}; show_normals::Bool = true, size = (600, 600)) where {P <: AbstractPanel, T}
    fig = Figure(size = size)
    ax = Axis(fig[1, 1], aspect = DataAspect())

    viz_2d!(ax, interface; show_normals = show_normals)

    return fig
end

function viz_3d!(
    ax::Axis3,
    interface::DielectricInterface{P, T};
    show_normals::Bool = false,
    show_points::Bool = true,
    highlight_panel::Union{Nothing, Int} = nothing,
    neighbor_list::Union{Nothing, Dict{Tuple{Int, Int}, Int}} = nothing,
    neighbor_max_order::Union{Nothing, Int} = nothing,
    neighbor_atol::Union{Nothing, T} = nothing,
    highlight_edges::Bool = false,
    highlight_color = :orange,
    neighbor_color = :green,
    edge_color = :cyan,
    base_color = :blue,
    fill_highlight::Bool = true,
    fill_neighbors::Bool = true,
    fill_alpha = 0.3,
) where {P <: AbstractPanel, T}
    t = 0.2
    neighbor_indices = Set{Int}()
    if highlight_panel !== nothing
        n_panels = length(interface.panels)
        (1 <= highlight_panel <= n_panels) || throw(ArgumentError("highlight_panel must be between 1 and $(n_panels)"))
        if neighbor_list === nothing
            if neighbor_max_order === nothing || neighbor_atol === nothing
                throw(ArgumentError("highlight_panel requires neighbor_list or neighbor_max_order and neighbor_atol"))
            end
            neighbor_list = build_neighbor_list(interface, neighbor_max_order, neighbor_atol, true, true)
        end
        for ((i, j), _) in neighbor_list
            i == highlight_panel || continue
            push!(neighbor_indices, j)
        end
    end

    gb = GLMakie.GeometryBasics
    nan_pt = gb.Point3f(NaN, NaN, NaN)

    line_groups = Dict{Symbol, Vector{gb.Point3f}}(
        :base => gb.Point3f[],
        :highlight => gb.Point3f[],
        :neighbor => gb.Point3f[],
        :edge => gb.Point3f[],
    )
    point_groups = Dict{Symbol, Vector{gb.Point3f}}(
        :base => gb.Point3f[],
        :highlight => gb.Point3f[],
        :neighbor => gb.Point3f[],
        :edge => gb.Point3f[],
    )
    normal_segments = gb.Point3f[]

    fill_vertices_highlight = gb.Point3f[]
    fill_faces_highlight = gb.TriangleFace[]
    fill_vertices_neighbor = gb.Point3f[]
    fill_faces_neighbor = gb.TriangleFace[]

    function panel_group(panel_idx, panel)
        if highlight_panel !== nothing
            if panel_idx == highlight_panel
                return :highlight
            elseif panel_idx in neighbor_indices
                return :neighbor
            elseif highlight_edges && panel.is_edge
                return :edge
            else
                return :base
            end
        elseif highlight_edges && panel.is_edge
            return :edge
        else
            return :base
        end
    end

    function append_quad!(pts, a, b, c, d)
        push!(pts, gb.Point3f(a[1], a[2], a[3]))
        push!(pts, gb.Point3f(b[1], b[2], b[3]))
        push!(pts, gb.Point3f(c[1], c[2], c[3]))
        push!(pts, gb.Point3f(d[1], d[2], d[3]))
        push!(pts, gb.Point3f(a[1], a[2], a[3]))
        push!(pts, nan_pt)
    end

    function append_fill!(verts, faces, a, b, c, d)
        base_idx = length(verts)
        push!(verts, gb.Point3f(a[1], a[2], a[3]))
        push!(verts, gb.Point3f(b[1], b[2], b[3]))
        push!(verts, gb.Point3f(c[1], c[2], c[3]))
        push!(verts, gb.Point3f(d[1], d[2], d[3]))
        push!(faces, gb.TriangleFace(base_idx + 1, base_idx + 2, base_idx + 3))
        push!(faces, gb.TriangleFace(base_idx + 1, base_idx + 3, base_idx + 4))
    end

    for (panel_idx, panel) in enumerate(interface.panels)
        a, b, c, d = panel.corners
        group = panel_group(panel_idx, panel)
        append_quad!(line_groups[group], a, b, c, d)

        if fill_highlight && highlight_panel !== nothing
            should_fill = panel_idx == highlight_panel || (fill_neighbors && panel_idx in neighbor_indices)
            if should_fill
                if panel_idx == highlight_panel
                    append_fill!(fill_vertices_highlight, fill_faces_highlight, a, b, c, d)
                elseif panel_idx in neighbor_indices
                    append_fill!(fill_vertices_neighbor, fill_faces_neighbor, a, b, c, d)
                end
            end
        end

        if show_points
            for p in panel.points
                push!(point_groups[group], gb.Point3f(p[1], p[2], p[3]))
            end
        end

        if show_normals
            nx, ny, nz = t .* panel.normal
            for p in panel.points
                p0 = gb.Point3f(p[1], p[2], p[3])
                p1 = gb.Point3f(p[1] + nx, p[2] + ny, p[3] + nz)
                push!(normal_segments, p0)
                push!(normal_segments, p1)
            end
        end
    end

    if !isempty(fill_faces_highlight)
        mesh!(ax, gb.Mesh(fill_vertices_highlight, fill_faces_highlight), color = (highlight_color, fill_alpha), transparency = true)
    end
    if !isempty(fill_faces_neighbor)
        mesh!(ax, gb.Mesh(fill_vertices_neighbor, fill_faces_neighbor), color = (neighbor_color, fill_alpha), transparency = true)
    end

    if !isempty(line_groups[:base])
        lines!(ax, line_groups[:base], color = base_color, linewidth = 0.6)
    end
    if !isempty(line_groups[:highlight])
        lines!(ax, line_groups[:highlight], color = highlight_color, linewidth = 0.6)
    end
    if !isempty(line_groups[:neighbor])
        lines!(ax, line_groups[:neighbor], color = neighbor_color, linewidth = 0.6)
    end
    if !isempty(line_groups[:edge])
        lines!(ax, line_groups[:edge], color = edge_color, linewidth = 0.6)
    end

    if show_points
        if !isempty(point_groups[:base])
            scatter!(ax, point_groups[:base], color = :red, markersize = 3)
        end
        if !isempty(point_groups[:highlight])
            scatter!(ax, point_groups[:highlight], color = highlight_color, markersize = 3)
        end
        if !isempty(point_groups[:neighbor])
            scatter!(ax, point_groups[:neighbor], color = neighbor_color, markersize = 3)
        end
        if !isempty(point_groups[:edge])
            scatter!(ax, point_groups[:edge], color = edge_color, markersize = 3)
        end
    end

    if show_normals && !isempty(normal_segments)
        linesegments!(ax, normal_segments, color = :black, linewidth = 0.4)
    end
end

function viz_3d(
    interface::DielectricInterface{P, T};
    show_normals::Bool = false,
    show_points::Bool = true,
    highlight_panel::Union{Nothing, Int} = nothing,
    neighbor_list::Union{Nothing, Dict{Tuple{Int, Int}, Int}} = nothing,
    neighbor_max_order::Union{Nothing, Int} = nothing,
    neighbor_atol::Union{Nothing, T} = nothing,
    highlight_edges::Bool = false,
    highlight_color = :orange,
    neighbor_color = :green,
    edge_color = :cyan,
    base_color = :blue,
    fill_highlight::Bool = true,
    fill_neighbors::Bool = true,
    fill_alpha = 0.3,
    size = (700, 600),
) where {P <: AbstractPanel, T}
    fig = Figure(size = size)
    ax = Axis3(fig[1, 1], aspect = :data)
    viz_3d!(
        ax,
        interface;
        show_normals = show_normals,
        show_points = show_points,
        highlight_panel = highlight_panel,
        neighbor_list = neighbor_list,
        neighbor_max_order = neighbor_max_order,
        neighbor_atol = neighbor_atol,
        highlight_edges = highlight_edges,
        highlight_color = highlight_color,
        neighbor_color = neighbor_color,
        edge_color = edge_color,
        base_color = base_color,
        fill_highlight = fill_highlight,
        fill_neighbors = fill_neighbors,
        fill_alpha = fill_alpha,
    )
    return fig
end

function viz_3d(
    interfaces::Vector{<:DielectricInterface{P, T}};
    show_normals::Bool = false,
    show_points::Bool = true,
    highlight_panel::Union{Nothing, Int} = nothing,
    neighbor_list::Union{Nothing, Dict{Tuple{Int, Int}, Int}} = nothing,
    neighbor_max_order::Union{Nothing, Int} = nothing,
    neighbor_atol::Union{Nothing, T} = nothing,
    highlight_edges::Bool = false,
    highlight_color = :orange,
    neighbor_color = :green,
    edge_color = :cyan,
    base_color = :blue,
    fill_highlight::Bool = true,
    fill_neighbors::Bool = true,
    fill_alpha = 0.6,
    size = (700, 600),
) where {P <: AbstractPanel, T}
    fig = Figure(size = size)
    ax = Axis3(fig[1, 1], aspect = :data)
    for interface in interfaces
        viz_3d!(
            ax,
            interface;
            show_normals = show_normals,
            show_points = show_points,
            highlight_panel = highlight_panel,
            neighbor_list = neighbor_list,
            neighbor_max_order = neighbor_max_order,
            neighbor_atol = neighbor_atol,
            highlight_edges = highlight_edges,
            highlight_color = highlight_color,
            neighbor_color = neighbor_color,
            edge_color = edge_color,
            base_color = base_color,
            fill_highlight = fill_highlight,
            fill_neighbors = fill_neighbors,
            fill_alpha = fill_alpha,
        )
    end
    return fig
end

function viz_3d!(
    ax::Axis3,
    source::VolumeSource{T, 3};
    colorrange::Union{Nothing, Tuple{T, T}} = nothing,
    min_density::T = zero(T),
    markersize::Real = 6,
    alpha::Real = 0.8,
    algorithm = :mip,
) where {T}
    xs, ys, zs = source.axes
    dens = source.density
    if !(BoundaryIntegral._is_uniform_axis(xs) && BoundaryIntegral._is_uniform_axis(ys) && BoundaryIntegral._is_uniform_axis(zs))
        (xs, ys, zs), dens = _resample_volume_to_uniform((xs, ys, zs), dens)
    end

    colormap = to_colormap(:plasma)
    colormap[1] = RGBAf(0,0,0,0)

    volume!(
        ax,
        (first(xs), last(xs)),
        (first(ys), last(ys)),
        (first(zs), last(zs)),
        dens;
        colormap = colormap,
        # algorithm = algorithm,
        # absorption=4f0
    )
    return ax
end

function viz_3d(
    source::VolumeSource{T, 3};
    colorrange::Union{Nothing, Tuple{T, T}} = nothing,
    min_density::T = zero(T),
    markersize::Real = 6,
    alpha::Real = 0.8,
    algorithm = :mip,
    size = (700, 600),
) where {T}
    fig = Figure(size = size)
    ax = Axis3(fig[1, 1], aspect = :data)
    viz_3d!(
        ax,
        source;
        colorrange = colorrange,
        min_density = min_density,
        markersize = markersize,
        alpha = alpha,
        algorithm = algorithm,
    )
    return fig
end

function viz_3d(;
    interfaces = nothing,
    sources = nothing,
    show_normals::Bool = false,
    show_points::Bool = true,
    highlight_panel::Union{Nothing, Int} = nothing,
    neighbor_list::Union{Nothing, Dict{Tuple{Int, Int}, Int}} = nothing,
    neighbor_max_order = nothing,
    neighbor_atol = nothing,
    highlight_edges::Bool = false,
    highlight_color = :orange,
    neighbor_color = :green,
    edge_color = :cyan,
    base_color = :blue,
    fill_highlight::Bool = true,
    fill_neighbors::Bool = true,
    fill_alpha = 0.3,
    colorrange = nothing,
    min_density = zero(Float64),
    markersize::Real = 6,
    alpha::Real = 0.8,
    algorithm = :mip,
    size = (700, 600),
)
    fig = Figure(size = size)
    ax = Axis3(fig[1, 1], aspect = :data)

    if interfaces !== nothing
        for interface in (interfaces isa DielectricInterface ? (interfaces,) : interfaces)
            viz_3d!(
                ax,
                interface;
                show_normals = show_normals,
                show_points = show_points,
                highlight_panel = highlight_panel,
                neighbor_list = neighbor_list,
                neighbor_max_order = neighbor_max_order,
                neighbor_atol = neighbor_atol,
                highlight_edges = highlight_edges,
                highlight_color = highlight_color,
                neighbor_color = neighbor_color,
                edge_color = edge_color,
                base_color = base_color,
                fill_highlight = fill_highlight,
                fill_neighbors = fill_neighbors,
                fill_alpha = fill_alpha,
            )
        end
    end

    if sources !== nothing
        for source in (sources isa VolumeSource ? (sources,) : sources)
            viz_3d!(
                ax,
                source;
                colorrange = colorrange,
                min_density = min_density,
                markersize = markersize,
                alpha = alpha,
                algorithm = algorithm,
            )
        end
    end

    return fig
end

end
