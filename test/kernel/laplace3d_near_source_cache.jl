using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature, LinearAlgebra
using Test

# Pin the flatten convention used by `_laplace3d_corrections`:
# Dw[i, j], output K_up[ti, idx] with idx = (ii - 1) * n_quad + jj (outer ii).
# A SourceCache built from the same panel must reproduce the same K_up matrix
# bit-for-bit at the BLAS-3 level (modulo floating-point summation order).
# Note on shapes: `interp_matrix_1d_gl(ns0, ws0, ns_up)` returns an Ex of shape
# (n_up × n_quad) — query points along the first axis, source nodes along the
# second. The Mt formula below indexes it accordingly.
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

    @test norm(K_ref) > 1e-3   # guard against the test passing on near-zero values
    @test isapprox(K_new, K_ref; rtol = 1e-12, atol = 1e-14)
end

# Independent check that the scale * ws_up * ws_up factors in Mt are placed
# correctly. Pick a plain "kernel" function kvec[α] = f(p_up[α]); compute
# K_block[m] = Mt * kvec, and compare to the explicit weighted sum
#   K_block[m] = Σ_{i,j} scale * ws_up[i] * ws_up[j] * kvec[α] * Ex[i,m_x] * Ex[j,m_y].
# Both formulations must agree; if Mt had wrong weights or scale, this would fail.
@testset "Mt weights and scale factors" begin
    n_quad = 3
    n_up   = 7
    ns0, ws0 = gausslegendre(n_quad);  ns0 = Float64.(ns0); ws0 = Float64.(ws0)
    ns_up, ws_up = gausslegendre(n_up); ns_up = Float64.(ns_up); ws_up = Float64.(ws_up)
    Ex = BI.interp_matrix_1d_gl(ns0, ws0, ns_up)

    a = (-0.7, -0.4, 0.2); b = ( 0.5, -0.4, 0.2)
    c = ( 0.5,  0.6, 0.2); d = (-0.7,  0.6, 0.2)
    Lx = norm(b .- a); Ly = norm(d .- a); scale = Lx * Ly / 4

    # Build Mt per spec §5.3.
    Mt = Matrix{Float64}(undef, n_quad^2, n_up^2)
    for m_x in 1:n_quad, m_y in 1:n_quad
        m = (m_x - 1) * n_quad + m_y
        for i_up in 1:n_up, j_up in 1:n_up
            α = (i_up - 1) * n_up + j_up
            Mt[m, α] = scale * ws_up[i_up] * ws_up[j_up] *
                       Ex[i_up, m_x] * Ex[j_up, m_y]
        end
    end

    # Smooth synthetic "kernel" values at upsampled positions (factors do NOT cancel).
    kvec = Vector{Float64}(undef, n_up^2)
    for i_up in 1:n_up, j_up in 1:n_up
        α = (i_up - 1) * n_up + j_up
        u = ns_up[i_up]; v = ns_up[j_up]
        kvec[α] = cos(1.7 * u) * sin(0.9 * v) + 0.3
    end

    K_new = Mt * kvec

    # Explicit weighted-sum reference.
    K_ref = zeros(Float64, n_quad^2)
    for m_x in 1:n_quad, m_y in 1:n_quad
        m = (m_x - 1) * n_quad + m_y
        s = 0.0
        for i_up in 1:n_up, j_up in 1:n_up
            α = (i_up - 1) * n_up + j_up
            s += scale * ws_up[i_up] * ws_up[j_up] *
                 kvec[α] * Ex[i_up, m_x] * Ex[j_up, m_y]
        end
        K_ref[m] = s
    end

    @test norm(K_ref) > 1e-2
    @test isapprox(K_new, K_ref; rtol = 1e-12, atol = 1e-14)
end

@testset "build_source_cache shapes and reuse" begin
    ns0, ws0 = gausslegendre(4); ns0 = Float64.(ns0); ws0 = Float64.(ws0)
    a = (-0.5, -0.5, 0.0); b = ( 0.5, -0.5, 0.0)
    c = ( 0.5,  0.5, 0.0); d = (-0.5,  0.5, 0.0)
    normal = (0.0, 0.0, 1.0)
    panel = BI.rect_panel3d_discretize(a, b, c, d, ns0, ws0, normal)

    n_up = 10
    cache = BI.build_source_cache(panel, n_up)
    @test cache.panel === panel
    @test cache.n_up == n_up
    @test length(cache.p_up) == n_up^2
    @test size(cache.Mt) == (panel.n_quad^2, n_up^2)

    # Equivalence to current laplace3d_panel_upsampled output, sample target:
    a2 = (-0.6, -0.6, 0.4); b2 = ( 0.6, -0.6, 0.4)
    c2 = ( 0.6,  0.6, 0.4); d2 = (-0.6,  0.6, 0.4)
    panel_trg = BI.rect_panel3d_discretize(a2, b2, c2, d2, ns0, ws0, normal)

    K_old = BI.laplace3d_DT_panel_upsampled(panel, panel_trg, n_up)
    # New path: for each target, build kvec and apply Mt.
    K_new = similar(K_old)
    np_trg = size(K_new, 1)
    kvec = Vector{Float64}(undef, n_up^2)
    for t in 1:np_trg
        pt = panel_trg.points[t]
        for α in 1:n_up^2
            kvec[α] = BI.laplace3d_grad(cache.p_up[α], pt, panel_trg.normal)
        end
        K_new[t, :] .= cache.Mt * kvec
    end

    @test isapprox(K_new, K_old; rtol = 1e-12, atol = 1e-14)
end
