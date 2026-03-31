using Test
using BoundaryIntegral
using LinearAlgebra
const BI = BoundaryIntegral

@testset "laplace2d_near_singular" begin
    # Build a singular source panel (GJ quadrature) with exponent = 0.5 (gamma-1)
    n_quad = 8
    exponent = 0.5
    a_src = (0.0, 0.0)
    b_src = (0.1, 0.0)
    normal_src = (0.0, 1.0)
    panel_src = BI.line_panel2d_singular_discretize(a_src, b_src, n_quad, exponent, normal_src)

    # Build a regular target panel nearby (offset in normal direction so correction is nonzero)
    ns_gl, ws_gl = BI.gausslegendre(n_quad)
    a_trg = (0.0, 0.05)
    b_trg = (0.1, 0.05)
    normal_trg = (0.0, 1.0)
    panel_trg = BI.line_panel2d_discretize(a_trg, b_trg, ns_gl, ws_gl, normal_trg)

    @testset "block dimensions and finiteness" begin
        A_near, A_direct = BI.laplace2d_near_singular_block(panel_src, panel_trg)

        @test size(A_near) == (n_quad, n_quad)
        @test size(A_direct) == (n_quad, n_quad)
        @test all(isfinite, A_near)
        @test all(isfinite, A_direct)
    end

    @testset "correction is nonzero" begin
        A_near, A_direct = BI.laplace2d_near_singular_block(panel_src, panel_trg)
        delta = A_near .- A_direct
        @test norm(delta) > 0
    end

    @testset "interface corrections" begin
        # Build a small box with singular panels
        box = BI.single_dielectric_box2d(1.0, 1.0, 8, 0.2, 0.05, 5.0, 1.0, Float64; use_singular=true)
        corrections = BI.laplace2d_near_singular_corrections(box; range_factor=3.0)

        @test length(corrections) > 0
        for (delta_A, src_range, trg_range) in corrections
            @test size(delta_A) == (length(trg_range), length(src_range))
            @test all(isfinite, delta_A)
        end
    end
end
