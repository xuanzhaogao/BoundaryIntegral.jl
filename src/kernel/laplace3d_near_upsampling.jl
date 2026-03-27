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

function laplace3d_DT_corrections(interface::DielectricInterface{P, T}, neighbor_list::Dict{Tuple{Int, Int}, Int}) where {P <: AbstractPanel, T}

    cnt = zeros(Int, length(interface.panels))
    for i in 1:length(interface.panels)
        cnt[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    rows = Int[]
    cols = Int[]
    vals = T[]

    for ((i, j), n_up) in neighbor_list
        panel_src = interface.panels[i]
        panel_trg = interface.panels[j]

        DT_up = laplace3d_DT_panel_upsampled(panel_src, panel_trg, n_up)
        DT_direct = laplace3d_DT_panel(panel_src, panel_trg)
        block = DT_up - DT_direct

        row_range = (offsets[j] + 1):offsets[j + 1]
        col_range = (offsets[i] + 1):offsets[i + 1]

        for (r_local, r) in enumerate(row_range)
            for (c_local, c) in enumerate(col_range)
                v = block[r_local, c_local]
                iszero(v) && continue
                push!(rows, r)
                push!(cols, c)
                push!(vals, v)
            end
        end
    end

    return sparse(rows, cols, vals, total_n, total_n)
end

function laplace3d_D_corrections(interface::DielectricInterface{P, T}, neighbor_list::Dict{Tuple{Int, Int}, Int}) where {P <: AbstractPanel, T}

    cnt = zeros(Int, length(interface.panels))
    for i in 1:length(interface.panels)
        cnt[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    rows = Int[]
    cols = Int[]
    vals = T[]

    for ((i, j), n_up) in neighbor_list
        panel_src = interface.panels[i]
        panel_trg = interface.panels[j]

        D_up = laplace3d_D_panel_upsampled(panel_src, panel_trg, n_up)
        D_direct = laplace3d_D_panel(panel_src, panel_trg)
        block = D_up - D_direct

        row_range = (offsets[j] + 1):offsets[j + 1]
        col_range = (offsets[i] + 1):offsets[i + 1]

        for (r_local, r) in enumerate(row_range)
            for (c_local, c) in enumerate(col_range)
                v = block[r_local, c_local]
                iszero(v) && continue
                push!(rows, r)
                push!(cols, c)
                push!(vals, v)
            end
        end
    end

    return sparse(rows, cols, vals, total_n, total_n)
end
