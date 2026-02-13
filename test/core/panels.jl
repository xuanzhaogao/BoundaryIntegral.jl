using BoundaryIntegral
using Test

@testset "interface uniform grid sampling" begin
    interface = BoundaryIntegral.single_dielectric_box3d(1.0, 1.0, 1.0, 3, 0.3, 2.0, 1.0, Float64)
    sigma = [point.panel_point.point[1] + 2.0 * point.panel_point.point[2] - 0.5 * point.panel_point.point[3]
             for point in BoundaryIntegral.eachpoint(interface)]

    surfaces = BoundaryIntegral.interface_uniform_samples(interface, sigma; n_sample = 4)

    @test length(surfaces) == 6
    for s in surfaces
        @test size(s.X) == (4, 4)
        @test size(s.Y) == (4, 4)
        @test size(s.Z) == (4, 4)
        @test size(s.V) == (4, 4)
        for j in 1:4, i in 1:4
            p = (s.X[i, j], s.Y[i, j], s.Z[i, j])
            expected = p[1] + 2.0 * p[2] - 0.5 * p[3]
            @test s.V[i, j] ≈ expected atol = 1e-8
        end
    end
end
