using BoundaryIntegral
import BoundaryIntegral as BI
using Test

@testset "AdaptiveConfig defaults" begin
    cfg = BI.AdaptiveConfig(atol = 1e-8)
    @test cfg.atol == 1e-8
    @test cfg.rtol == sqrt(eps(Float64))
    @test cfg.n_GL == 0
    @test cfg.max_depth == 20

    cfg2 = BI.AdaptiveConfig(atol = 1e-6, rtol = 1e-10, n_GL = 8, max_depth = 12)
    @test cfg2.atol == 1e-6
    @test cfg2.rtol == 1e-10
    @test cfg2.n_GL == 8
    @test cfg2.max_depth == 12
end

using HCubature, FastGaussQuadrature, LinearAlgebra, StaticArrays

@testset "adaptive_panel_moments_inplace! matches HCubature" begin
    n_quad = 4
    ns, ws = gausslegendre(n_quad); ns = Float64.(ns); ws = Float64.(ws)

    # Unit square panel at z = 0.
    a = (-0.5, -0.5, 0.0); b = ( 0.5, -0.5, 0.0)
    c = ( 0.5,  0.5, 0.0); d = (-0.5,  0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    panel = BI.rect_panel3d_discretize(a, b, c, d, ns, ws, normal)

    # Target slightly above the panel (smooth, not singular) so HCubature converges easily.
    pt = (0.1, 0.2, 0.3)
    pt_normal = (0.0, 0.0, 1.0)

    cfg = BI.AdaptiveConfig(atol = 1e-10, rtol = 1e-12, n_GL = n_quad, max_depth = 12)
    K_row = zeros(Float64, n_quad^2)
    BI.adaptive_panel_moments_inplace!(K_row, panel, pt, pt_normal, :DT, cfg)

    # Reference moments via HCubature, integrand in (u,v) ∈ [-1,1]^2.
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a; dma = d .- a
    Lx = norm(bma); Ly = norm(dma); scale_panel = Lx * Ly / 4
    K_ref = zeros(Float64, n_quad^2)
    for m_x in 1:n_quad, m_y in 1:n_quad
        m = (m_x - 1) * n_quad + m_y
        integrand(uv) = begin
            u, v = uv[1], uv[2]
            y = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            rx = BI.barycentric_row(panel.gl_xs, panel.bary_weights, u)
            ry = BI.barycentric_row(panel.gl_xs, panel.bary_weights, v)
            return BI.laplace3d_grad(y, pt, pt_normal) * rx[m_x] * ry[m_y] * scale_panel
        end
        val, _ = hcubature(integrand, SVector{2,Float64}(-1.0, -1.0),
                                          SVector{2,Float64}( 1.0,  1.0); atol = 1e-12)
        K_ref[m] = val
    end

    @test isapprox(K_row, K_ref; atol = 1e-7, rtol = 1e-7)
end
