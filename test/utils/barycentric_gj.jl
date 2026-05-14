using Test
using BoundaryIntegral
using FastGaussQuadrature
using LinearAlgebra
const BI_GJ = BoundaryIntegral

@testset "gj_barycentric_weights interpolation" begin
    n = 8
    x, _ = gaussjacobi(n, -0.3, 0.0)
    lambda = BI_GJ.gj_barycentric_weights(x)

    # Interpolation of a degree-(n-1) polynomial must be exact
    f = x .^ (n - 1)
    xq = 0.123456789
    r = zeros(Float64, n)
    BI_GJ.barycentric_row!(r, x, lambda, xq)
    interp_val = dot(r, f)
    exact_val = xq^(n - 1)

    @test abs(interp_val - exact_val) < 1e-12
end
