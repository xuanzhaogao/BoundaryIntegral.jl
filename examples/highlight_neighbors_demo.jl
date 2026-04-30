using BoundaryIntegral
using BoundaryIntegral: build_neighbor_list, AbstractPanel, DielectricInterface
using LinearAlgebra
using CairoMakie

Lx, Ly, Lz = 10.0, 10.0, 1.0
n_quad = 6
l_ec = 0.5
eps_in, eps_out = 2.0, 1.0

interface = single_dielectric_box3d(Lx, Ly, Lz, n_quad, l_ec, eps_in, eps_out)
n_panels = length(interface.panels)
println("panels: ", n_panels)

max_order = 30
atol = 1e-6
neighbor_list = build_neighbor_list(interface, max_order, atol, true, true)
println("neighbor pairs total: ", length(neighbor_list))

panel_size(p) = norm(p.corners[2] .- p.corners[1]) * norm(p.corners[3] .- p.corners[2])

groups = [Int[] for _ in 1:n_panels]
for ((i, j), ord) in neighbor_list
    push!(groups[i], ord)
end

# Pick a "large" interior panel (max area, not on edge).
non_edge = [i for (i, p) in enumerate(interface.panels) if !p.is_edge]
sizes = [panel_size(interface.panels[i]) for i in non_edge]
panel_large = non_edge[argmax(sizes)]

println("large interior -> panel #", panel_large,
        "  size=", round(panel_size(interface.panels[panel_large]); digits=3),
        "  #neighbors=", length(groups[panel_large]),
        "  orders=", sort!(unique(groups[panel_large])))

all_orders = collect(values(neighbor_list))
shared_cr = (Float32(minimum(all_orders)), Float32(maximum(all_orders)))

fig = viz_3d(
    interface;
    highlight_panel = panel_large,
    neighbor_list = neighbor_list,
    show_points = false,
    fill_alpha = 0.6,
    neighbor_colorrange = shared_cr,
    add_neighbor_colorbar = false,
    size = (800, 800),
)
ax = fig.content[1]
ax.azimuth = -π/2
ax.elevation = π/2

out = joinpath(@__DIR__, "highlight_neighbors_demo.png")
save(out, fig)
println("saved: ", out)
