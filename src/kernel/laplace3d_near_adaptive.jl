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

# Walk the adaptive quadtree and record the leaf cells (the 4 sub-cells at each
# stop node) without accumulating into a K_row. The tree topology is driven by a
# single representative target point.
#
# Returns (leaves::Vector{NTuple{4,T}}, hit_max::Bool) where each leaf is
# (u_lo, u_hi, v_lo, v_hi) — one of the four sub-cells that would have been
# summed into K_row at a stop node.
function walk_adaptive_tree(
    panel_src::FlatPanel{T,3},
    point_trg::NTuple{3,T},
    trg_normal::NTuple{3,T},
    mode::Symbol,
    cfg::AdaptiveConfig,
) where T
    n_quad = panel_src.n_quad
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

    n_m = n_quad^2
    # Preallocate all buffers once; cell_moments_inplace! fills them in-place.
    rx    = Vector{T}(undef, n_quad)
    ry    = Vector{T}(undef, n_quad)
    parent_buf   = Vector{T}(undef, n_m)
    c1_buf       = Vector{T}(undef, n_m)
    c2_buf       = Vector{T}(undef, n_m)
    c3_buf       = Vector{T}(undef, n_m)
    c4_buf       = Vector{T}(undef, n_m)
    children_buf = Vector{T}(undef, n_m)

    function cell_moments_inplace!(out::Vector{T}, u_lo::T, u_hi::T, v_lo::T, v_hi::T)
        fill!(out, zero(T))
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
                    error("unknown mode for walk_adaptive_tree")
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
    end

    leaves = NTuple{4,T}[]
    hit_max_flag = Ref(false)

    function recurse_walk!(u_lo::T, u_hi::T, v_lo::T, v_hi::T, depth::Int)
        cell_moments_inplace!(parent_buf, u_lo, u_hi, v_lo, v_hi)
        mid_u = (u_hi + u_lo) / 2
        mid_v = (v_hi + v_lo) / 2
        cell_moments_inplace!(c1_buf, u_lo, mid_u, v_lo, mid_v)
        cell_moments_inplace!(c2_buf, mid_u, u_hi, v_lo, mid_v)
        cell_moments_inplace!(c3_buf, u_lo, mid_u, mid_v, v_hi)
        cell_moments_inplace!(c4_buf, mid_u, u_hi, mid_v, v_hi)
        # children_sum = c1 + c2 + c3 + c4; compute in-place
        @inbounds for m in 1:n_m
            children_buf[m] = c1_buf[m] + c2_buf[m] + c3_buf[m] + c4_buf[m]
        end
        err = maximum(abs(parent_buf[m] - children_buf[m]) for m in 1:n_m)
        tol_local = max(T(cfg.atol), T(cfg.rtol) * maximum(abs(parent_buf[m]) for m in 1:n_m))
        if err < tol_local || depth == cfg.max_depth
            # Record the 4 sub-cells as leaves.
            push!(leaves, (u_lo, mid_u, v_lo, mid_v))
            push!(leaves, (mid_u, u_hi, v_lo, mid_v))
            push!(leaves, (u_lo, mid_u, mid_v, v_hi))
            push!(leaves, (mid_u, u_hi, mid_v, v_hi))
            if depth == cfg.max_depth && err >= tol_local
                hit_max_flag[] = true
            end
            return
        end
        recurse_walk!(u_lo, mid_u, v_lo, mid_v, depth + 1)
        recurse_walk!(mid_u, u_hi, v_lo, mid_v, depth + 1)
        recurse_walk!(u_lo, mid_u, mid_v, v_hi, depth + 1)
        recurse_walk!(mid_u, u_hi, mid_v, v_hi, depth + 1)
    end

    recurse_walk!(-one(T), one(T), -one(T), one(T), 0)
    return leaves, hit_max_flag[]
end

# Per-(source panel, target panel) cache for the adaptive correction path.
# Mirrors SourceCache but uses the adaptive leaf-cell distribution determined
# by a representative target point rather than a fixed upsampling grid.
struct SourceAdaptiveCache{T}
    panel::FlatPanel{T,3}
    n_GL::Int
    leaves::Vector{NTuple{4,T}}   # (u_lo, u_hi, v_lo, v_hi) per leaf cell
    p_adp::Vector{NTuple{3,T}}    # length = length(leaves) * n_GL^2
    Mt::Matrix{T}                 # (n_quad^2 × length(p_adp))
end

function build_adaptive_source_cache(
    panel_src::FlatPanel{T,3},
    point_trg_rep::NTuple{3,T},
    trg_normal_rep::NTuple{3,T},
    mode::Symbol,
    cfg::AdaptiveConfig,
) where T
    leaves, hit_max = walk_adaptive_tree(panel_src, point_trg_rep, trg_normal_rep, mode, cfg)

    n_quad = panel_src.n_quad
    n_GL = cfg.n_GL == 0 ? n_quad : cfg.n_GL
    ns_d, ws_d = gausslegendre(n_GL)
    ns_GL = convert(Vector{T}, ns_d)
    ws_GL = convert(Vector{T}, ws_d)

    a, b, c, d = panel_src.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a
    Lx = norm(bma); Ly = norm(dma); scale_panel = Lx * Ly / 4

    gl_xs = panel_src.gl_xs
    bary_weights = panel_src.bary_weights

    n_leaves = length(leaves)
    n_adp = n_leaves * n_GL^2
    p_adp = Vector{NTuple{3,T}}(undef, n_adp)
    Mt = Matrix{T}(undef, n_quad^2, n_adp)

    rx = Vector{T}(undef, n_quad)
    ry = Vector{T}(undef, n_quad)

    for (ℓ, leaf) in enumerate(leaves)
        u_lo, u_hi, v_lo, v_hi = leaf
        half_u = (u_hi - u_lo) / 2
        half_v = (v_hi - v_lo) / 2
        mid_u  = (u_hi + u_lo) / 2
        mid_v  = (v_hi + v_lo) / 2
        scale_cell = half_u * half_v * scale_panel

        for gi in 1:n_GL
            u = mid_u + half_u * ns_GL[gi]
            wu = ws_GL[gi]
            barycentric_row!(rx, gl_xs, bary_weights, u)
            for gj in 1:n_GL
                v = mid_v + half_v * ns_GL[gj]
                wv = ws_GL[gj]
                barycentric_row!(ry, gl_xs, bary_weights, v)

                α = (ℓ - 1) * n_GL^2 + (gi - 1) * n_GL + gj
                p_adp[α] = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                w_α = scale_cell * wu * wv
                for m_x in 1:n_quad
                    for m_y in 1:n_quad
                        m = (m_x - 1) * n_quad + m_y
                        Mt[m, α] = w_α * rx[m_x] * ry[m_y]
                    end
                end
            end
        end
    end

    return SourceAdaptiveCache{T}(panel_src, n_GL, leaves, p_adp, Mt), hit_max
end

# Compute K_block[t, m] = ∫_panel K(x, trg_t) L_m(x) J dx for all targets simultaneously.
# Uses the leaf list from walk_adaptive_tree (driven by a representative target) to
# amortize tree-traversal cost, then evaluates the kernel for all np_trg targets per
# GL quadrature point using tight loops (no large position cache needed).
#
# K_block must be pre-allocated as (np_trg, n_quad^2), zeroed before calling.
function adaptive_block_moments_inplace!(
    K_block::AbstractMatrix{T},
    panel_src::FlatPanel{T,3},
    trg_points::AbstractVector{NTuple{3,T}},
    trg_normal::NTuple{3,T},
    leaves::Vector{NTuple{4,T}},
    mode::Symbol,
    cfg::AdaptiveConfig,
) where T
    n_quad = panel_src.n_quad
    np_trg = length(trg_points)
    @assert size(K_block) == (np_trg, n_quad^2)

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
    kvals = Vector{T}(undef, np_trg)

    for leaf in leaves
        u_lo, u_hi, v_lo, v_hi = leaf
        half_u = (u_hi - u_lo) / 2
        half_v = (v_hi - v_lo) / 2
        mid_u  = (u_hi + u_lo) / 2
        mid_v  = (v_hi + v_lo) / 2
        scale_cell = half_u * half_v * scale_panel

        for gi in 1:n_GL
            u = mid_u + half_u * ns_GL[gi]
            wu = ws_GL[gi]
            barycentric_row!(rx, gl_xs, bary_weights, u)
            for gj in 1:n_GL
                v = mid_v + half_v * ns_GL[gj]
                wv = ws_GL[gj]
                barycentric_row!(ry, gl_xs, bary_weights, v)

                y = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                w_uv = wu * wv * scale_cell

                # Evaluate kernel for all targets — tight loop, potentially vectorizable.
                if mode === :DT
                    @inbounds for t in 1:np_trg
                        kvals[t] = laplace3d_grad(y, trg_points[t], trg_normal)
                    end
                elseif mode === :D
                    @inbounds for t in 1:np_trg
                        kvals[t] = laplace3d_grad(trg_points[t], y, src_normal)
                    end
                else
                    error("unknown mode for adaptive_block_moments_inplace!")
                end

                # Accumulate into K_block[t, m] for all targets and moments.
                @inbounds for m_x in 1:n_quad
                    rx_mx = rx[m_x]
                    for m_y in 1:n_quad
                        m = (m_x - 1) * n_quad + m_y
                        w_m = w_uv * rx_mx * ry[m_y]
                        for t in 1:np_trg
                            K_block[t, m] += w_m * kvals[t]
                        end
                    end
                end
            end
        end
    end
end
