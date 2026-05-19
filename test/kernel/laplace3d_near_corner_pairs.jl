using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature
using Test

# Two perpendicular panels sharing a single edge along x = 0.5.
# panel_a lies in the z = 0 plane; panel_b lies in the x = 0.5 plane.
# With low n_quad their GL nodes are interior, so a node-only KDTree
# may not discover them; the panel-corner phase must.
@testset "build_neighbor_list corner-pair discovery (correct_edges)" begin
    n_quad = 3
    ns, ws = gausslegendre(n_quad); ns = Float64.(ns); ws = Float64.(ws)

    a1 = (-0.5, -0.5, 0.0); b1 = ( 0.5, -0.5, 0.0)
    c1 = ( 0.5,  0.5, 0.0); d1 = (-0.5,  0.5, 0.0)
    normal1 = (0.0, 0.0, 1.0)
    panel_a = BI.rect_panel3d_discretize(a1, b1, c1, d1, ns, ws, normal1)

    # panel_b shares the edge x = 0.5, y in [-0.5, 0.5] with panel_a.
    a2 = ( 0.5, -0.5, 0.0); b2 = ( 0.5,  0.5, 0.0)
    c2 = ( 0.5,  0.5, 1.0); d2 = ( 0.5, -0.5, 1.0)
    normal2 = (1.0, 0.0, 0.0)
    panel_b = BI.rect_panel3d_discretize(a2, b2, c2, d2, ns, ws, normal2)

    interface = BI.DielectricInterface([panel_a, panel_b], [1.0, 1.0], [1.0, 1.0])

    # With correct_edges = false: classic discovery; range_factor is small
    # enough that the two interior-node clouds do not see each other.
    (; upsample, adaptive) =
        BI.build_neighbor_list(interface, 1, 1e-6;
                               distance_only = true, range_factor = 0.5,
                               correct_edges = false)
    @test isempty(adaptive)

    # With correct_edges = true: the corner KDTree must surface the touching pair.
    (; upsample, adaptive) =
        BI.build_neighbor_list(interface, 1, 1e-6;
                               distance_only = true, range_factor = 0.5,
                               correct_edges = true)
    @test haskey(adaptive, (1, 2)) || haskey(adaptive, (2, 1))
end
