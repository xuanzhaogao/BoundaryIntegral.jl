using FastGaussQuadrature
using LinearAlgebra

include("../../src/utils/barycentric.jl")

# Generate Gauss-Jacobi nodes (alpha=-0.3, beta=0.0) with n=8
n = 8
x, _ = gaussjacobi(n, -0.3, 0.0)

# Compute barycentric weights for these nodes
λ = gj_barycentric_weights(x)

# Test: interpolation of a degree-(n-1) polynomial must be exact.
# Use f(t) = t^(n-1) as the test function.
f = x .^ (n - 1)

# Evaluate the interpolant at a non-node point
xq = 0.123456789
r = zeros(Float64, n)
barycentric_row!(r, x, λ, xq)
interp_val = dot(r, f)
exact_val  = xq^(n - 1)

err = abs(interp_val - exact_val)
println("Barycentric interpolation of x^$(n-1) at xq=$xq")
println("  interpolated: $interp_val")
println("  exact:        $exact_val")
println("  error:        $err")

tol = 1e-12
@assert err < tol "Interpolation error $err exceeds tolerance $tol"
println("Test passed.")
