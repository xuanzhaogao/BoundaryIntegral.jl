using BoundaryIntegral
using FastGaussQuadrature
using Random
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

function _eval_poly(coeffs::AbstractVector{T}, x::T) where T
    val = zero(T)
    @inbounds for k in reverse(eachindex(coeffs))
        val = muladd(val, x, coeffs[k])
    end
    return val
end

@testset "interface_approx reproduces high-order polynomial data" begin
    Random.seed!(1)
    n_quad = 32
    ns, ws = gausslegendre(n_quad)
    ns = Float64.(ns)
    ws = Float64.(ws)

    panel = BoundaryIntegral.rect_panel3d_discretize(
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        ns,
        ws,
        (0.0, 0.0, 1.0),
    )
    interface = BoundaryIntegral.DielectricInterface([panel], [2.0], [1.0])

    coeffx = randn(n_quad)
    coeffy = randn(n_quad)
    vals = [
        _eval_poly(coeffx, point[1]) + _eval_poly(coeffy, point[2])
        for point in panel.points
    ]
    approx = BoundaryIntegral.interface_approx(interface, vals)

    worst = 0.0
    for _ in 1:200
        u = -1 + 2 * rand()
        v = -1 + 2 * rand()
        exact = _eval_poly(coeffx, u) + _eval_poly(coeffy, v)
        worst = max(worst, abs(approx((u, v, 0.0)) - exact))
    end

    @test worst < 1e-9
end
