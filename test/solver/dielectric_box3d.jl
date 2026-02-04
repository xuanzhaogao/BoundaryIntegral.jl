using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra, Krylov
using Random
using Test

@testset "dielectric_box3d" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(1.2, 0.8, 0.6, 4, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))

    lhs = BI.Lhs_dielectric_box3d(interface)
    lhs_fmm3d = BI.Lhs_dielectric_box3d_fmm3d(interface, 1e-12)
    rhs = BI.Rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(interface)

    x = BI.solve_lu(lhs, rhs)
    @test norm(lhs * x - rhs) < 1e-10

    x_trial = randn(BI.num_points(interface))
    @test norm(lhs * x_trial - lhs_fmm3d * x_trial) < 1e-8

    x_gmres = BI.solve_gmres(lhs_fmm3d, rhs, 1e-12, 1e-12)
    @test norm(lhs_fmm3d * x_gmres - rhs) < 1e-10

    total_flux = dot(ws, x)
    @test isapprox(total_flux + 1.0 / eps_box, 1.0, atol = 1e-1)

    total_flux_gmres = dot(ws, x_gmres)
    @test isapprox(total_flux_gmres + 1.0 / eps_box, 1.0, atol = 1e-1)
end

@testset "laplace3d_pottrg_near near/far" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 2, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))
    sol = randn(BI.num_points(interface))

    # far away target
    target = (100.0, 100.0, 100.0)
    pot_direct = (BI.laplace3d_pottrg(interface, reshape(collect(target), 3, 1)) * sol)[1]
    pot_hc_1 = BI.laplace3d_pottrg_near(interface, target, sol, 1e-8; range_factor = 0.0)
    pot_hc_2 = BI.laplace3d_pottrg_near(interface, target, sol, 1e-8; range_factor = Inf)
    @test isapprox(pot_hc_1, pot_direct, rtol = 1e-8)
    @test isapprox(pot_hc_2, pot_direct, rtol = 1e-8)

    # near target
    target = (0.6, 0.1, 0.2)
    pot_direct = (BI.laplace3d_pottrg(interface, reshape(collect(target), 3, 1)) * sol)[1]
    pot_hc_1 = BI.laplace3d_pottrg_near(interface, target, sol, 1e-6; range_factor = 5.0)
    pot_hc_2 = BI.laplace3d_pottrg_near(interface, target, sol, 1e-6; range_factor = Inf)
    # @test isapprox(pot_hc_1, pot_direct, rtol = 0.15)
    @test isapprox(pot_hc_1, pot_hc_2, atol = 1e-4)
end

@testset "dielectric_box3d corrected" begin
    eps_box = 4.0
    interface = BI.single_dielectric_box3d(3.0, 3.0, 1.0, 6, 0.2, eps_box, 1.0, Float64; alpha = sqrt(2))

    lhs_uncorrected = BI.Lhs_dielectric_box3d_fmm3d(interface, 1e-6)
    lhs_corrected = BI.Lhs_dielectric_box3d_fmm3d_corrected(interface, 1e-6, 1e-6, 12)
    rhs = BI.Rhs_dielectric_box3d(interface, BI.PointSource((0.1, 0.1, 0.1), 1.0), eps_box)
    ws = BI.all_weights(interface)

    x_gmres_corrected = BI.solve_gmres(lhs_corrected, rhs, 1e-6, 1e-6)
    x_gmres_uncorrected = BI.solve_gmres(lhs_uncorrected, rhs, 1e-6, 1e-6)
    @test norm(lhs_corrected * x_gmres_corrected - rhs) < 1e-5
    @test norm(lhs_uncorrected * x_gmres_uncorrected - rhs) < 1e-5

    total_flux_corrected = dot(ws, x_gmres_corrected)
    total_flux_uncorrected = dot(ws, x_gmres_uncorrected)
    @test isapprox(total_flux_uncorrected + 1.0 / eps_box, 1.0, atol = 1e-1)
    @test isapprox(total_flux_corrected + 1.0 / eps_box, 1.0, atol = 1e-1)
end

@testset "dielectric_box3d rhs adaptive" begin
    eps_box = 4.0
    ps = BI.PointSource((0.1, 0.1, 0.1), 1.0)
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.2,
        0.8,
        0.6,
        2,
        ps,
        eps_box,
        0.099,
        1e-8,
        eps_box,
        1.0,
        Float64;
        max_depth = 2,
    )
    lhs = BI.Lhs_dielectric_box3d(interface)
    rhs = BI.Rhs_dielectric_box3d(interface, ps, eps_box)
    x = BI.solve_gmres(lhs, rhs, 1e-6, 1e-6)
    @test norm(lhs * x - rhs) < 1e-5
    @test abs(dot(BI.all_weights(interface), x) +  1.0 / eps_box - 1.0) < 1e-2

    ps = BI.PointSource((1.0, 1.0, 1.0), 1.0)
    interface = BI.single_dielectric_box3d_rhs_adaptive(
        1.2,
        0.8,
        0.6,
        2,
        ps,
        1.0,
        0.099,
        1e-8,
        eps_box,
        1.0,
        Float64;
        max_depth = 2,
    )
    lhs = BI.Lhs_dielectric_box3d(interface)
    rhs = BI.Rhs_dielectric_box3d(interface, ps, 1.0)
    x = BI.solve_gmres(lhs, rhs, 1e-6, 1e-6)
    @test norm(lhs * x - rhs) < 1e-5
    @test abs(dot(BI.all_weights(interface), x)) < 1e-2
end
