using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature
using Test

@testset "laplace3d_near smoke" begin
    ns, ws = gausslegendre(2)
    ns = Float64.(ns)
    ws = Float64.(ws)

    normal = (0.0, 0.0, 1.0)
    p1 = BI.rect_panel3d_discretize(
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        ns,
        ws,
        normal,
    )
    p2 = BI.rect_panel3d_discretize(
        (2.0, 2.0, 0.0),
        (2.2, 2.0, 0.0),
        (2.2, 2.2, 0.0),
        (2.0, 2.2, 0.0),
        ns,
        ws,
        normal,
    )
    interface = BI.DielectricInterface([p1, p2], fill(2.0, 2), fill(1.0, 2))

    targets = reshape([0.02, -0.01, 0.03], 3, 1)
    op = BI.laplace3d_pottrg_fmm3d_corrected_hcubature(interface, targets, 1e-10, 1e-9, 5.0)

    sigma = [cos(0.1 * i) for i in 1:BI.num_points(interface)]
    vals = op * sigma
    @test isfinite(vals[1])
end
