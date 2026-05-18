struct SourceCache{T}
    panel::FlatPanel{T,3}
    n_up::Int
    p_up::Vector{NTuple{3,T}}    # length n_up^2, upsampled physical positions
    Mt::Matrix{T}                # n_quad^2 x n_up^2 moments-to-nodal tensor
end

function build_source_cache(panel::FlatPanel{T,3}, n_up::Int) where T
    ns_up_d, ws_up_d = gausslegendre(n_up)
    ns_up = convert(Vector{T}, ns_up_d)
    ws_up = convert(Vector{T}, ws_up_d)

    # Per-source Ex from THIS panel's GL nodes (handles varquad).
    Ex = convert(Matrix{T}, interp_matrix_1d_gl(panel.gl_xs, panel.gl_ws, ns_up))

    a, b, c, d = panel.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    Lx = norm(b .- a)
    Ly = norm(d .- a)
    scale = Lx * Ly / 4
    bma = b .- a
    dma = d .- a

    p_up = Vector{NTuple{3,T}}(undef, n_up^2)
    for i_up in 1:n_up
        x = ns_up[i_up] / 2
        for j_up in 1:n_up
            y = ns_up[j_up] / 2
            alpha = (i_up - 1) * n_up + j_up
            p_up[alpha] = cc .+ bma .* x .+ dma .* y
        end
    end

    n_quad = panel.n_quad
    Mt = Matrix{T}(undef, n_quad^2, n_up^2)
    for m_x in 1:n_quad
        for m_y in 1:n_quad
            m = (m_x - 1) * n_quad + m_y
            for i_up in 1:n_up
                for j_up in 1:n_up
                    alpha = (i_up - 1) * n_up + j_up
                    Mt[m, alpha] = scale * ws_up[i_up] * ws_up[j_up] *
                               Ex[i_up, m_x] * Ex[j_up, m_y]
                end
            end
        end
    end

    return SourceCache{T}(panel, n_up, p_up, Mt)
end

function laplace3d_panel_upsampled(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}, n_up::Int, mode::Symbol) where T
    ns_up, ws_up = gausslegendre(n_up)
    ns_up = T.(ns_up)
    ws_up = T.(ws_up)

    ns0 = panel_src.gl_xs
    ws0 = panel_src.gl_ws

    Ex = interp_matrix_1d_gl(ns0, ws0, ns_up)
    Ey = Ex

    a, b, c, d = panel_src.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    Lx = norm(b .- a)
    Ly = norm(d .- a)
    scale = Lx * Ly / 4

    bma = b .- a
    dma = d .- a
    n_quad = panel_src.n_quad
    np_trg = num_points(panel_trg)
    K_up = zeros(T, np_trg, n_quad * n_quad)
    nthreads = Base.Threads.maxthreadid()
    D_weighted = [Matrix{T}(undef, n_up, n_up) for _ in 1:nthreads]
    temp = [Matrix{T}(undef, n_quad, n_up) for _ in 1:nthreads]
    block = [Matrix{T}(undef, n_quad, n_quad) for _ in 1:nthreads]
    points_trg = panel_trg.points

    Base.Threads.@threads for ti in 1:np_trg
        tid = Base.Threads.threadid()
        D_weighted_tid = D_weighted[tid]
        temp_tid = temp[tid]
        block_tid = block[tid]
        point_trg = points_trg[ti]
        @inbounds for j in 1:n_up
            y = ns_up[j] / 2
            wy = ws_up[j]
            for i in 1:n_up
                x = ns_up[i] / 2
                p = cc .+ bma .* x .+ dma .* y
                if mode === :DT
                    D_weighted_tid[i, j] = laplace3d_grad(p, point_trg, panel_trg.normal) * (ws_up[i] * wy * scale)
                elseif mode === :D
                    D_weighted_tid[i, j] = laplace3d_grad(point_trg, p, panel_src.normal) * (ws_up[i] * wy * scale)
                else
                    error("unknown mode for laplace3d_panel_upsampled")
                end
            end
        end
        mul!(temp_tid, transpose(Ex), D_weighted_tid)
        mul!(block_tid, temp_tid, Ey)
        # Match panel point ordering (ii outer, jj inner) from rect_panel3d_discretize.
        idx = 1
        @inbounds for ii in 1:n_quad
            for jj in 1:n_quad
                K_up[ti, idx] = block_tid[ii, jj]
                idx += 1
            end
        end
    end

    return K_up
end

# this function generate a block of the correction matrix
function laplace3d_DT_panel_upsampled(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}, n_up::Int) where T
    return laplace3d_panel_upsampled(panel_src, panel_trg, n_up, :DT)
end

function laplace3d_D_panel_upsampled(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}, n_up::Int) where T
    return laplace3d_panel_upsampled(panel_src, panel_trg, n_up, :D)
end

# Shared body for DT/D corrections. mode is :DT or :D; direct_kernel selects
# the un-corrected per-pair direct evaluation.
function _laplace3d_corrections(
    interface::DielectricInterface{P, T},
    neighbor_list::Dict{Tuple{Int, Int}, Int},
    mode::Symbol, direct_kernel::Function,
) where {P <: AbstractPanel, T}
    cnt = [length(p.points) for p in interface.panels]
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    # Group neighbors by source panel index.
    src_to_neighbors = Dict{Int, Vector{Tuple{Int, Int}}}()   # i => [(j, n_up), ...]
    for ((i, j), n_up) in neighbor_list
        push!(get!(() -> Vector{Tuple{Int,Int}}(), src_to_neighbors, i), (j, n_up))
    end
    source_indices = collect(keys(src_to_neighbors))

    nthreads = Base.Threads.maxthreadid()
    rows_tl = [Int[] for _ in 1:nthreads]
    cols_tl = [Int[] for _ in 1:nthreads]
    vals_tl = [T[]   for _ in 1:nthreads]

    Base.Threads.@threads :dynamic for k in 1:length(source_indices)
        i = source_indices[k]
        panel_src = interface.panels[i]
        neighbors = src_to_neighbors[i]
        n_up_max = maximum(nup for (_, nup) in neighbors)

        cache = build_source_cache(panel_src, n_up_max)   # one per source panel

        tid = Base.Threads.threadid()
        rows = rows_tl[tid]
        cols = cols_tl[tid]
        vals = vals_tl[tid]
        col_off = offsets[i]
        ncols = offsets[i + 1] - col_off

        # The per-pair n_up is intentionally ignored: each source panel uses
        # n_up_max across all its neighbors so the SourceCache is built once.
        for (j, _n_up) in neighbors
            panel_trg = interface.panels[j]
            np_trg = num_points(panel_trg)

            # Build Kmat[alpha, t] over all targets in panel j (BLAS-3 friendly).
            n_up_eff = cache.n_up                # we use the panel's full n_up_max cache
            Kmat = Matrix{T}(undef, n_up_eff^2, np_trg)
            if mode === :DT
                for t in 1:np_trg
                    pt = panel_trg.points[t]
                    for alpha in 1:(n_up_eff^2)
                        Kmat[alpha, t] = laplace3d_grad(cache.p_up[alpha], pt, panel_trg.normal)
                    end
                end
            elseif mode === :D
                for t in 1:np_trg
                    pt = panel_trg.points[t]
                    for alpha in 1:(n_up_eff^2)
                        Kmat[alpha, t] = laplace3d_grad(pt, cache.p_up[alpha], panel_src.normal)
                    end
                end
            else
                error("unknown mode for _laplace3d_corrections")
            end

            K_block_T = cache.Mt * Kmat            # (n_quad^2, np_trg)
            K_block = transpose(K_block_T)         # (np_trg, n_quad^2)

            K_direct = direct_kernel(panel_src, panel_trg)

            row_off = offsets[j]
            nrows = offsets[j + 1] - row_off
            @inbounds for c_local in 1:ncols
                for r_local in 1:nrows
                    v = K_block[r_local, c_local] - K_direct[r_local, c_local]
                    iszero(v) && continue
                    push!(rows, row_off + r_local)
                    push!(cols, col_off + c_local)
                    push!(vals, v)
                end
            end
        end
    end

    return sparse(reduce(vcat, rows_tl), reduce(vcat, cols_tl),
                  reduce(vcat, vals_tl), total_n, total_n)
end

function laplace3d_DT_corrections(interface::DielectricInterface{P, T},
                                  neighbor_list::Dict{Tuple{Int, Int}, Int}) where {P <: AbstractPanel, T}
    return _laplace3d_corrections(interface, neighbor_list, :DT, laplace3d_DT_panel)
end

function laplace3d_D_corrections(interface::DielectricInterface{P, T},
                                 neighbor_list::Dict{Tuple{Int, Int}, Int}) where {P <: AbstractPanel, T}
    return _laplace3d_corrections(interface, neighbor_list, :D, laplace3d_D_panel)
end
