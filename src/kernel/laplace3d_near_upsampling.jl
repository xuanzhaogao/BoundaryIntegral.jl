struct SourceCache{T}
    panel::FlatPanel{T,3}
    n_up::Int
    p_up::Vector{NTuple{3,T}}    # length n_up^2, upsampled physical positions
    Mt::Matrix{T}                # n_quad^2 × n_up^2 moments-to-nodal tensor
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
            α = (i_up - 1) * n_up + j_up
            p_up[α] = cc .+ bma .* x .+ dma .* y
        end
    end

    n_quad = panel.n_quad
    Mt = Matrix{T}(undef, n_quad^2, n_up^2)
    for m_x in 1:n_quad
        for m_y in 1:n_quad
            m = (m_x - 1) * n_quad + m_y
            for i_up in 1:n_up
                for j_up in 1:n_up
                    α = (i_up - 1) * n_up + j_up
                    Mt[m, α] = scale * ws_up[i_up] * ws_up[j_up] *
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

# Internal helper: same math as laplace3d_panel_upsampled, but writes into
# caller-supplied scratch buffers (allocated once per worker thread) and has
# no internal threading. Used by the pair-parallel corrections builder.
# Buffers must be sized for the maximum n_up seen across the neighbor list:
#   D_weighted: n_up_max × n_up_max
#   temp_buf  : n_quad   × n_up_max
#   block_buf : n_quad   × n_quad
# Returns a freshly-allocated `K_up` (np_trg × n_quad²); we cannot reuse a
# scratch K_up across pairs because each correction block has a different
# target panel size in general.
function _laplace3d_panel_upsampled_inplace!(
    panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3},
    n_up::Int, ns_up::AbstractVector{T}, ws_up::AbstractVector{T},
    Ex::AbstractMatrix{T}, mode::Symbol,
    D_weighted::AbstractMatrix{T}, temp_buf::AbstractMatrix{T},
    block_buf::AbstractMatrix{T},
) where T
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
    Dw = @view D_weighted[1:n_up, 1:n_up]
    tb = @view temp_buf[1:n_quad, 1:n_up]
    bb = @view block_buf[1:n_quad, 1:n_quad]
    Exv = @view Ex[1:n_up, 1:n_quad]
    Eyv = Exv
    points_trg = panel_trg.points

    @inbounds for ti in 1:np_trg
        point_trg = points_trg[ti]
        for j in 1:n_up
            y = ns_up[j] / 2
            wy = ws_up[j]
            for i in 1:n_up
                x = ns_up[i] / 2
                p = cc .+ bma .* x .+ dma .* y
                if mode === :DT
                    Dw[i, j] = laplace3d_grad(p, point_trg, panel_trg.normal) * (ws_up[i] * wy * scale)
                elseif mode === :D
                    Dw[i, j] = laplace3d_grad(point_trg, p, panel_src.normal) * (ws_up[i] * wy * scale)
                else
                    error("unknown mode for _laplace3d_panel_upsampled_inplace!")
                end
            end
        end
        mul!(tb, transpose(Exv), Dw)
        mul!(bb, tb, Eyv)
        idx = 1
        for ii in 1:n_quad
            for jj in 1:n_quad
                K_up[ti, idx] = bb[ii, jj]
                idx += 1
            end
        end
    end
    return K_up
end

# Build {n_up => (ns_up, ws_up, Ex)} once per call. Assumes every source panel
# in interface uses the same (gl_xs, gl_ws) — true for the canonical Nyström
# discretization where p_quad is fixed across the interface.
function _build_upsampling_cache(interface::DielectricInterface{P, T},
                                 neighbor_list::Dict{Tuple{Int, Int}, Int}) where {P <: AbstractPanel, T}
    ref = interface.panels[1]
    ns0 = ref.gl_xs
    ws0 = ref.gl_ws
    distinct = Set(values(neighbor_list))
    cache = Dict{Int, Tuple{Vector{T}, Vector{T}, Matrix{T}}}()
    for n_up in distinct
        ns_d, ws_d = gausslegendre(n_up)
        ns_up = convert(Vector{T}, ns_d)
        ws_up = convert(Vector{T}, ws_d)
        Ex = convert(Matrix{T}, interp_matrix_1d_gl(ns0, ws0, ns_up))
        cache[n_up] = (ns_up, ws_up, Ex)
    end
    return cache
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

    cache = _build_upsampling_cache(interface, neighbor_list)
    pairs = collect(neighbor_list)

    n_up_max = isempty(pairs) ? 0 : maximum(p -> p.second, pairs)
    n_quad_max = isempty(interface.panels) ? 0 : maximum(p -> p.n_quad, interface.panels)

    nthreads = Base.Threads.maxthreadid()
    rows_tl = [Int[] for _ in 1:nthreads]
    cols_tl = [Int[] for _ in 1:nthreads]
    vals_tl = [T[]   for _ in 1:nthreads]
    # Per-thread scratch sized for the largest n_up seen in this call.
    Dw_tl    = [Matrix{T}(undef, n_up_max,   n_up_max)   for _ in 1:nthreads]
    tb_tl    = [Matrix{T}(undef, n_quad_max, n_up_max)   for _ in 1:nthreads]
    bb_tl    = [Matrix{T}(undef, n_quad_max, n_quad_max) for _ in 1:nthreads]

    Base.Threads.@threads :static for k in 1:length(pairs)
        (i, j), n_up = pairs[k]
        panel_src = interface.panels[i]
        panel_trg = interface.panels[j]
        ns_up, ws_up, Ex = cache[n_up]

        tid = Base.Threads.threadid()
        K_up = _laplace3d_panel_upsampled_inplace!(
            panel_src, panel_trg, n_up, ns_up, ws_up, Ex, mode,
            Dw_tl[tid], tb_tl[tid], bb_tl[tid])
        K_direct = direct_kernel(panel_src, panel_trg)

        rows = rows_tl[tid]
        cols = cols_tl[tid]
        vals = vals_tl[tid]
        row_off = offsets[j]
        col_off = offsets[i]
        nrows = offsets[j + 1] - row_off
        ncols = offsets[i + 1] - col_off
        @inbounds for c_local in 1:ncols
            for r_local in 1:nrows
                v = K_up[r_local, c_local] - K_direct[r_local, c_local]
                iszero(v) && continue
                push!(rows, row_off + r_local)
                push!(cols, col_off + c_local)
                push!(vals, v)
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
