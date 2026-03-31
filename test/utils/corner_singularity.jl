using Test
using BoundaryIntegral

@testset "corner_singularity_power" begin
    # Known value: right-angle box corner, eps_in=2, eps_out=1
    gamma = corner_singularity_power(pi / 2, 2.0, 1.0)
    @test gamma ≈ 1.1066007580762274 rtol = 1e-8

    # Trivial case: same permittivity => no singularity, gamma = 1
    @test corner_singularity_power(pi / 2, 1.0, 1.0) == 1.0
    @test corner_singularity_power(pi / 3, 5.0, 5.0) == 1.0

    # High contrast: eps_in << eps_out => gamma < 1 (singular density)
    gamma_high = corner_singularity_power(pi / 2, 1.0, 100.0)
    @test gamma_high < 1.0
    @test gamma_high > 0.0

    # Symmetry: swapping eps_in/eps_out with complementary angle gives same power
    gamma_swap = corner_singularity_power(2pi - pi / 2, 1.0, 2.0)
    @test gamma_swap ≈ 1.1066007580762274 rtol = 1e-8
end

@testset "corner_singularity_power_multi" begin
    # Two-material case via multi: should match the two-material convenience function
    # For eps_in=2, eps_out=1, alpha=pi/2: angles=[pi/2], epsilons=[2.0, 1.0]
    gamma_multi = corner_singularity_power_multi([pi / 2], [2.0, 1.0])
    gamma_two = corner_singularity_power(pi / 2, 2.0, 1.0)
    # The multi version uses theta_ODE_det which combines both parities;
    # check it finds a root near the known value
    @test gamma_multi ≈ gamma_two rtol = 1e-6

    # Three-material junction: should return a positive power
    gamma_3 = corner_singularity_power_multi([pi / 2, pi / 2], [2.0, 5.0, 1.0])
    @test gamma_3 > 0.0
end

println("All corner_singularity tests passed.")
