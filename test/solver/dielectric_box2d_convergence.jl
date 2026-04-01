using BoundaryIntegral
import BoundaryIntegral as BI
using LinearAlgebra
using Test

@testset "Convergence: singular vs regular panels" begin
    # High-contrast dielectric with eps_out >> eps_in gives gamma < 1 (truly singular density).
    # The box interior has permittivity eps_in; the exterior has eps_out.
    # Source is near a corner to stress the singularity.
    eps_in = 1.0
    eps_out = 200.0
    eps_src = eps_in   # source is inside the box
    Lx = 1.0
    Ly = 1.0
    l_panel = 0.2
    # Use l_corner=0.1 so that corner panels are large enough for GJ to have impact.
    l_corner = 0.1
    ps = BI.PointSource((0.001, 0.001), 1.0)

    gamma = BI.corner_singularity_power(pi / 2, eps_in, eps_out)
    @assert gamma < 1.0 "Expected gamma < 1 for singular density"

    # Gauss's law: the total surface flux should be  1/eps_out - 1/eps_in
    expected_flux = 1.0 / eps_out - 1.0 / eps_in

    n_quads = [4, 8, 12, 16]
    errors_regular = Float64[]
    errors_singular = Float64[]

    println("=" ^ 60)
    println("Convergence test: singular vs regular corner panels")
    println("eps_in=$eps_in, eps_out=$eps_out, gamma=$(round(gamma, sigdigits=6))")
    println("Source at (0.001, 0.001), l_corner=$l_corner")
    println("Expected flux = $expected_flux")
    println("=" ^ 60)

    for n_quad in n_quads
        # Regular panels (Gauss-Legendre everywhere)
        box_reg = BI.single_dielectric_box2d(Lx, Ly, n_quad, l_panel, l_corner, eps_in, eps_out, Float64; use_singular=false)
        lhs_reg = BI.lhs_dielectric_box2d(box_reg)
        rhs_reg = BI.rhs_dielectric_box2d(box_reg, ps, eps_src)
        x_reg = BI.solve_lu(lhs_reg, rhs_reg)
        ws_reg = BI.all_weights(box_reg)
        err_reg = abs(dot(ws_reg, x_reg) - expected_flux)
        push!(errors_regular, err_reg)

        # Singular panels (Gauss-Jacobi on innermost corner panels)
        box_sing = BI.single_dielectric_box2d(Lx, Ly, n_quad, l_panel, l_corner, eps_in, eps_out, Float64; use_singular=true)
        lhs_sing = BI.lhs_dielectric_box2d(box_sing)
        rhs_sing = BI.rhs_dielectric_box2d(box_sing, ps, eps_src)
        x_sing = BI.solve_lu(lhs_sing, rhs_sing)
        ws_sing = BI.all_weights(box_sing)
        err_sing = abs(dot(ws_sing, x_sing) - expected_flux)
        push!(errors_singular, err_sing)

        println("n_quad=$(lpad(n_quad, 2))  regular=$(round(err_reg, sigdigits=4))  singular=$(round(err_sing, sigdigits=4))")
    end

    println("=" ^ 60)

    # At the highest quadrature order, singular panels should be more accurate
    # because GJ quadrature naturally resolves the corner density singularity.
    @test errors_singular[end] < errors_regular[end]
    println("Singular error ($(errors_singular[end])) < Regular error ($(errors_regular[end])) at n_quad=$(n_quads[end]): PASS")
end
