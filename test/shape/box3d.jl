using BoundaryIntegral
import BoundaryIntegral as BI
using Test

@testset "box3d rhs adaptive" begin
    rhs_const(p, n) = 1.0
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.0,
        1.0,
        1.0,
        2,
        rhs_const,
        2.0,
        1e-6,
        2.0,
        1.0,
        Float64;
        max_depth = 3,
    )
    @test length(interface.panels) == 6

    ps = BI.PointSource((0.1, 0.1, 0.1), 1.0)
    interface_ps = BI.single_dielectric_box3d_rhs_adaptive(
        1.0,
        1.0,
        1.0,
        2,
        ps,
        1.0,
        2.0,
        1e-6,
        2.0,
        1.0,
        Float64;
        max_depth = 2,
    )
    @test length(interface_ps.panels) >= 6

    rhs_poly(p, n) = p[1]^4 + p[2]^4 + p[3]^4
    interface_refined = BI.single_dielectric_box3d_rhs_adaptive(
        1.0,
        1.0,
        1.0,
        2,
        rhs_poly,
        0.3,
        1e-6,
        2.0,
        1.0,
        Float64;
        max_depth = 2,
    )
    @test length(interface_refined.panels) > 6

    interface_refined_ec = BI.single_dielectric_box3d_rhs_adaptive(
        1.0,
        1.0,
        1.0,
        2,
        rhs_const,
        0.2,
        1e-6,
        2.0,
        1.0,
        Float64;
        max_depth = 1,
    )
    @test length(interface_refined_ec.panels) > 6
end
