module GLMakieExt

using GLMakie
using BoundaryIntegral
using BoundaryIntegral: AbstractPanel, DielectricInterface, build_neighbor_list

import BoundaryIntegral: viz_2d, viz_3d

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
            neighbor_list = build_neighbor_list(interface, neighbor_max_order, neighbor_atol)
        end
        for ((i, j), _) in neighbor_list
            i == highlight_panel || continue
            push!(neighbor_indices, j)
        end
    end

    for (panel_idx, panel) in enumerate(interface.panels)
        a, b, c, d = panel.corners
        panel_color = base_color
        if highlight_panel !== nothing
            if panel_idx == highlight_panel
                panel_color = highlight_color
            elseif panel_idx in neighbor_indices
                panel_color = neighbor_color
            elseif highlight_edges && panel.is_edge
                panel_color = edge_color
            end
        elseif highlight_edges && panel.is_edge
            panel_color = edge_color
        end
        if fill_highlight && highlight_panel !== nothing
            should_fill = panel_idx == highlight_panel || (fill_neighbors && panel_idx in neighbor_indices)
            if should_fill
                fill_color = panel_idx == highlight_panel ? highlight_color : neighbor_color
                gb = GLMakie.GeometryBasics
                points = gb.Point3f[(a[1], a[2], a[3]), (b[1], b[2], b[3]), (c[1], c[2], c[3]), (d[1], d[2], d[3])]
                faces = gb.TriangleFace[(1, 2, 3), (1, 3, 4)]
                mesh!(ax, gb.Mesh(points, faces), color = (fill_color, fill_alpha), transparency = true)
            end
        end
        lines!(ax, [a[1], b[1], c[1], d[1], a[1]], [a[2], b[2], c[2], d[2], a[2]], [a[3], b[3], c[3], d[3], a[3]], color = panel_color, linewidth = 0.6)

        if show_points
            xs = [p[1] for p in panel.points]
            ys = [p[2] for p in panel.points]
            zs = [p[3] for p in panel.points]
            point_color = (panel_idx == highlight_panel || panel_idx in neighbor_indices || (highlight_edges && panel.is_edge)) ? panel_color : :red
            scatter!(ax, xs, ys, zs, color = point_color, markersize = 3)
        end

        show_normals || continue
        nx, ny, nz = t .* panel.normal
        xs = [p[1] for p in panel.points]
        ys = [p[2] for p in panel.points]
        zs = [p[3] for p in panel.points]
        lines!(ax, xs .+ nx, ys .+ ny, zs .+ nz, color = :black, linewidth = 0.4)
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

end
