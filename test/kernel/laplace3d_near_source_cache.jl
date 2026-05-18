using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature, LinearAlgebra
using Test

# Pin the flatten convention used by `_laplace3d_panel_upsampled_inplace!`:
# Dw[i, j], output K_up[ti, idx] with idx = (ii - 1) * n_quad + jj (outer ii).
# A SourceCache built from the same panel must reproduce the same K_up matrix
# bit-for-bit at the BLAS-3 level (modulo floating-point summation order).
@testset "Mt indexing convention vs transpose(Ex)*Dw*Ex" begin
    n_quad = 4
    n_up   = 8
    ns0, ws0 = gausslegendre(n_quad);  ns0 = Float64.(ns0); ws0 = Float64.(ws0)
    ns_up, ws_up = gausslegendre(n_up); ns_up = Float64.(ns_up); ws_up = Float64.(ws_up)
    Ex = BI.interp_matrix_1d_gl(ns0, ws0, ns_up)

    # Build a unit-square source panel and a single off-panel target point
    a = (-0.5, -0.5, 0.0); b = ( 0.5, -0.5, 0.0)
    c = ( 0.5,  0.5, 0.0); d = (-0.5,  0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    panel = BI.rect_panel3d_discretize(a, b, c, d, ns0, ws0, normal)
    Lx = norm(b .- a); Ly = norm(d .- a); scale = Lx * Ly / 4

    # Synthetic Dw[i,j] (arbitrary values)
    Dw = [sin(i + 0.3 * j) for i in 1:n_up, j in 1:n_up]

    # Reference: existing code's transpose(Ex) * Dw * Ex flattened with
    # idx = (ii-1)*n_quad + jj   (ii outer, jj inner).
    bb_ref = transpose(Ex) * Dw * Ex
    K_ref = Vector{Float64}(undef, n_quad^2)
    idx = 1
    for ii in 1:n_quad, jj in 1:n_quad
        K_ref[idx] = bb_ref[ii, jj]
        idx += 1
    end

    # Mt formula per spec §5.3:
    # Mt[m, α] = scale * ws_up[i_up] * ws_up[j_up] * Ex[i_up, m_x] * Ex[j_up, m_y]
    # with α = (i_up-1)*n_up + j_up,  m = (m_x-1)*n_quad + m_y.
    Mt = Matrix{Float64}(undef, n_quad^2, n_up^2)
    for m_x in 1:n_quad, m_y in 1:n_quad
        m = (m_x - 1) * n_quad + m_y
        for i_up in 1:n_up, j_up in 1:n_up
            α = (i_up - 1) * n_up + j_up
            Mt[m, α] = scale * ws_up[i_up] * ws_up[j_up] *
                       Ex[i_up, m_x] * Ex[j_up, m_y]
        end
    end

    # Flatten Dw with the same α convention; this is the "kvec" that the
    # spec multiplies Mt by. But the regression here is against
    # transpose(Ex)*Dw*Ex which does NOT carry the `scale * ws*ws` factors,
    # so reconstruct the bare matmul as transpose(Ex) * Dwhat * Ex where
    # Dwhat[i,j] = Dw[i,j] / (scale * ws_up[i] * ws_up[j]) and Mt absorbs
    # those factors. Equivalent identity:
    Dwhat_vec = Vector{Float64}(undef, n_up^2)
    for i_up in 1:n_up, j_up in 1:n_up
        α = (i_up - 1) * n_up + j_up
        Dwhat_vec[α] = Dw[i_up, j_up] / (scale * ws_up[i_up] * ws_up[j_up])
    end
    K_new = Mt * Dwhat_vec

    @test isapprox(K_new, K_ref; rtol = 1e-12, atol = 1e-14)
end
