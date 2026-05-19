# Adaptive quadtree-based moments for touching (edge-sharing) near pairs.
# See docs/superpowers/specs/2026-05-18-near-correction-improvements-design.md §5.4.

struct AdaptiveConfig
    atol::Float64
    rtol::Float64
    n_GL::Int          # base GL order per leaf; 0 means "use source panel's n_quad"
    max_depth::Int
end

AdaptiveConfig(; atol::Float64, rtol::Float64 = sqrt(eps(Float64)),
                 n_GL::Int = 0, max_depth::Int = 20) =
    AdaptiveConfig(atol, rtol, n_GL, max_depth)

# Compute the n_quad^2 moments ∫ K(point_trg, X(u,v)) L_{m_x}(u) L_{m_y}(v) J du dv
# over the source panel, accumulated into K_row[m] with m = (m_x-1)*n_quad + m_y.
# Recursive quadtree on (u,v) ∈ [-1,1]^2; error-based stopping.
function adaptive_panel_moments_inplace!(
    K_row::AbstractVector{T},
    panel_src::FlatPanel{T,3},
    point_trg::NTuple{3,T},
    trg_normal::NTuple{3,T},
    mode::Symbol,
    cfg::AdaptiveConfig,
) where T
    n_quad = panel_src.n_quad
    @assert length(K_row) == n_quad^2
    fill!(K_row, zero(T))

    n_GL = cfg.n_GL == 0 ? n_quad : cfg.n_GL
    ns_d, ws_d = gausslegendre(n_GL)
    ns_GL = convert(Vector{T}, ns_d)
    ws_GL = convert(Vector{T}, ws_d)

    a, b, c, d = panel_src.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a
    Lx = norm(bma); Ly = norm(dma); scale_panel = Lx * Ly / 4

    src_normal = panel_src.normal
    gl_xs = panel_src.gl_xs
    bary_weights = panel_src.bary_weights

    rx = Vector{T}(undef, n_quad)
    ry = Vector{T}(undef, n_quad)

    function cell_moments(u_lo::T, u_hi::T, v_lo::T, v_hi::T)
        out = zeros(T, n_quad^2)
        half_u = (u_hi - u_lo) / 2
        half_v = (v_hi - v_lo) / 2
        mid_u  = (u_hi + u_lo) / 2
        mid_v  = (v_hi + v_lo) / 2
        scale_cell = half_u * half_v * scale_panel
        @inbounds for gi in 1:n_GL
            u = mid_u + half_u * ns_GL[gi]
            wu = ws_GL[gi]
            barycentric_row!(rx, gl_xs, bary_weights, u)
            for gj in 1:n_GL
                v = mid_v + half_v * ns_GL[gj]
                wv = ws_GL[gj]
                barycentric_row!(ry, gl_xs, bary_weights, v)

                y = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                kval = if mode === :DT
                    laplace3d_grad(y, point_trg, trg_normal)
                elseif mode === :D
                    laplace3d_grad(point_trg, y, src_normal)
                else
                    error("unknown mode for adaptive_panel_moments_inplace!")
                end
                w = wu * wv * scale_cell * kval
                m = 0
                for m_x in 1:n_quad
                    for m_y in 1:n_quad
                        m += 1
                        out[m] += w * rx[m_x] * ry[m_y]
                    end
                end
            end
        end
        return out
    end

    function recurse!(K_row::AbstractVector{T}, u_lo::T, u_hi::T, v_lo::T, v_hi::T, depth::Int)
        parent = cell_moments(u_lo, u_hi, v_lo, v_hi)
        mid_u = (u_hi + u_lo) / 2
        mid_v = (v_hi + v_lo) / 2
        c1 = cell_moments(u_lo, mid_u, v_lo, mid_v)
        c2 = cell_moments(mid_u, u_hi, v_lo, mid_v)
        c3 = cell_moments(u_lo, mid_u, mid_v, v_hi)
        c4 = cell_moments(mid_u, u_hi, mid_v, v_hi)
        children_sum = c1 .+ c2 .+ c3 .+ c4
        err = maximum(abs.(parent .- children_sum))
        tol_local = max(T(cfg.atol), T(cfg.rtol) * maximum(abs.(parent)))
        if err < tol_local || depth == cfg.max_depth
            @inbounds for m in eachindex(K_row)
                K_row[m] += children_sum[m]
            end
            return depth == cfg.max_depth && err >= tol_local
        end
        hit_a = recurse!(K_row, u_lo, mid_u, v_lo, mid_v, depth + 1)
        hit_b = recurse!(K_row, mid_u, u_hi, v_lo, mid_v, depth + 1)
        hit_c = recurse!(K_row, u_lo, mid_u, mid_v, v_hi, depth + 1)
        hit_d = recurse!(K_row, mid_u, u_hi, mid_v, v_hi, depth + 1)
        return hit_a || hit_b || hit_c || hit_d
    end

    hit_max = recurse!(K_row, -one(T), one(T), -one(T), one(T), 0)
    return hit_max
end
