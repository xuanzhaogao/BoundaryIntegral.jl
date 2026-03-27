using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra, Krylov
using Random
using Test

@testset "dielectric_box" begin
    eps_box = 5.0

    for Lx in [1.0, 2.05]
        for Ly in [1.0, 3.03]
            box = BI.single_dielectric_box2d(Lx, Ly, 8, 0.2, 0.05, 5.0, 1.0, Float64)
            lhs = BI.lhs_dielectric_box2d(box)
            lhs_fmm2d = BI.lhs_dielectric_box2d_fmm2d(box, 1e-12)
            rhs = BI.rhs_dielectric_box2d(box, BI.PointSource((0.1, 0.1), 1.0), eps_box)
            ws = BI.all_weights(box)

            x = BI.solve_lu(lhs, rhs)
            @test norm(lhs * x - rhs) < 1e-10

            total_flux = dot(ws, x)
            @test isapprox(total_flux + 1.0 / eps_box, 1.0, atol = 1e-3)

            x_gmres = BI.solve_gmres(lhs_fmm2d, rhs,1e-12, 1e-12)
            @test norm(lhs_fmm2d * x_gmres - rhs) < 1e-10

            total_flux_gmres = dot(ws, x_gmres)
            @test isapprox(total_flux_gmres + 1.0 / eps_box, 1.0, atol = 1e-3)
        end
    end
end

@testset "dielectric_box multi_box2d" begin
    rects_vec = [[BI.square(0.0, 0.0), BI.square(1.0, 0.0)], [BI.square(0.0, 0.0), BI.square(1.0, 0.0), BI.square(0.5, 1.0)]]
    epses_vec = [[2.0, 3.0], [2.0, 3.0, 4.0]]
    for (rects, epses) in zip(rects_vec, epses_vec)
        interface = BI.multi_dielectric_box2d(8, 0.2, 0.05, rects, epses)

        lhs = BI.lhs_dielectric_box2d(interface)
        lhs_fmm2d = BI.lhs_dielectric_box2d_fmm2d(interface, 1e-12)
        rhs = BI.rhs_dielectric_box2d(interface, BI.PointSource((0.1, 0.1), 1.0), epses[1])

        x_trial = randn(BI.num_points(interface))
        @test norm(lhs * x_trial - lhs_fmm2d * x_trial) < 1e-9

        x = BI.solve_lu(lhs, rhs)
        @test norm(lhs * x - rhs) < 1e-10

        total_flux = dot(BI.all_weights(interface), x)
        @test isapprox(total_flux + 1.0 / epses[1], 1.0, atol = 1e-3)
    end
end
