function laplace3d_DT_panel(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}) where T

    np_src = num_points(panel_src)
    np_trg = num_points(panel_trg)

    DT = zeros(T, np_trg, np_src)
    for (i, pointi) in enumerate(eachpoint(panel_src))
        for (j, pointj) in enumerate(eachpoint(panel_trg))
            DT[j, i] = laplace3d_grad(pointi.point, pointj.point, pointj.normal)
        end
    end
    return DT * diagm(panel_src.weights)
end

# this function generate a block of the correction matrix
function laplace3d_DT_panel_upsampled(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}, n_up::Int) where T
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
    DT_up = zeros(T, np_trg, n_quad * n_quad)
    nthreads = Base.Threads.maxthreadid()
    D_weighted = [Matrix{T}(undef, n_up, n_up) for _ in 1:nthreads]
    temp = [Matrix{T}(undef, n_quad, n_up) for _ in 1:nthreads]
    block = [Matrix{T}(undef, n_quad, n_quad) for _ in 1:nthreads]
    points_trg = panel_trg.points
    normal_trg = panel_trg.normal

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
                D_weighted_tid[i, j] = laplace3d_grad(p, point_trg, normal_trg) * (ws_up[i] * wy * scale)
            end
        end
        mul!(temp_tid, transpose(Ex), D_weighted_tid)
        mul!(block_tid, temp_tid, Ey)
        @views DT_up[ti, :] .= vec(block_tid)
    end

    return DT_up
end

function laplace3d_DT_panel_hcubature(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}, atol::T) where T
    ns = panel_src.gl_xs
    ws = panel_src.gl_ws
    λ = gl_barycentric_weights(ns, ws)
    a, b, c, d = panel_src.corners
    cc = (a .+ b .+ c .+ d) ./ 4
    bma = b .- a
    dma = d .- a
    Lx = norm(b .- a)
    Ly = norm(d .- a)
    scale = Lx * Ly / 4

    n_quad = panel_src.n_quad
    np_trg = num_points(panel_trg)
    DT_exact = zeros(T, np_trg, n_quad * n_quad)
    points_trg = panel_trg.points
    normal_trg = panel_trg.normal

    for ti in 1:np_trg
        point_trg = points_trg[ti]
        function integrand(x)
            u = x[1]
            v = x[2]
            rx = T.(barycentric_row(ns, λ, u))
            ry = T.(barycentric_row(ns, λ, v))
            p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            k = laplace3d_grad(p, point_trg, normal_trg) * scale
            vals = Vector{T}(undef, n_quad * n_quad)
            idx = 1
            for jj in 1:n_quad
                for ii in 1:n_quad
                    vals[idx] = k * rx[ii] * ry[jj]
                    idx += 1
                end
            end
            return vals
        end
        res, _ = hcubature(integrand, T[-1, -1], T[1, 1]; atol = atol)
        @views DT_exact[ti, :] .= res
    end

    return DT_exact
end

# neighbor list describes the pair of panels that are near each other and the order needed for evaluation
function build_neighbor_list(
    interface::DielectricInterface{P, T},
    max_order::Int,
    atol::T,
    include_edges_src::Bool,
    include_edges_trg::Bool;
    distance_only::Bool = false,
    range_factor::T = T(5),
) where {P <: AbstractPanel, T}
    neighbor_list = Dict{Tuple{Int, Int}, Int}()
    n_panels = length(interface.panels)
    centers = Matrix{T}(undef, 3, n_panels)
    lengths = Vector{T}(undef, n_panels)
    n_quads = Vector{Int}(undef, n_panels)
    normals = Vector{NTuple{3, T}}(undef, n_panels)
    plane_offsets = Vector{T}(undef, n_panels)

    for (i, panel) in enumerate(interface.panels)
        c_panel = (panel.corners[1] .+ panel.corners[2] .+ panel.corners[3] .+ panel.corners[4]) ./ 4
        @views centers[:, i] .= c_panel
        lengths[i] = max(norm(panel.corners[1] .- panel.corners[2]), norm(panel.corners[2] .- panel.corners[3]))
        n_quads[i] = panel.n_quad
        normals[i] = panel.normal
        plane_offsets[i] = dot(panel.normal, panel.corners[1])
    end

    n_points = sum(length(panel.points) for panel in interface.panels)
    points = Matrix{T}(undef, 3, n_points)
    point_panel_idx = Vector{Int}(undef, n_points)
    point_idx = 1
    for (panel_idx, panel) in enumerate(interface.panels)
        for point in panel.points
            @views points[:, point_idx] .= point
            point_panel_idx[point_idx] = panel_idx
            point_idx += 1
        end
    end
    tree = KDTree(points)

    same_surface_tol = sqrt(eps(T))

    for (i, paneli) in enumerate(interface.panels)
        (!include_edges_src && paneli.is_edge) && continue
        l_i = lengths[i]
        n_quad_i = n_quads[i]
        r_i = range_factor * l_i / n_quad_i
        nearby = inrange(tree, centers[:, i], r_i)

        panel_dict = Dict{Int, Vector{Int}}()
        for point_id in nearby
            j = point_panel_idx[point_id]
            i == j && continue
            if haskey(panel_dict, j)
                push!(panel_dict[j], point_id)
            else
                panel_dict[j] = [point_id]
            end
        end

        for j in keys(panel_dict)

            (!include_edges_trg && interface.panels[j].is_edge) && continue

            dot_normals = dot(normals[i], normals[j])
            if dot_normals > 1 - same_surface_tol
                if abs(plane_offsets[i] - plane_offsets[j]) <= same_surface_tol * max(one(T), l_i)
                    continue
                end
            end

            if distance_only
                neighbor_list[(i, j)] = n_quad_i
            else
                # find the closest point in panel j to panel i
                points_j = panel_dict[j]
                min_dist = Inf
                min_point_id = 0
                for point_id in points_j
                    dist = norm(points[:, point_id] - centers[:, i])
                    if dist < min_dist
                        min_dist = dist
                        min_point_id = point_id
                    end
                end

                order_i = check_quad_order3d(paneli, (points[1, min_point_id], points[2, min_point_id], points[3, min_point_id]), atol, max_order)

                if order_i > n_quad_i
                    key = (i, j)
                    if haskey(neighbor_list, key) 
                        neighbor_list[key] = max(neighbor_list[key], order_i)
                    else
                        neighbor_list[key] = order_i
                    end
                end
            end
        end
    end

    return neighbor_list
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

function laplace3d_DT_corrections_hcubature(
    interface::DielectricInterface{P, T},
    neighbor_list::Dict{Tuple{Int, Int}, Int},
    atol::T,
) where {P <: AbstractPanel, T}

    cnt = zeros(Int, length(interface.panels))
    for i in 1:length(interface.panels)
        cnt[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    rows = Int[]
    cols = Int[]
    vals = T[]

    for ((i, j), _) in neighbor_list
        panel_src = interface.panels[i]
        panel_trg = interface.panels[j]

        DT_exact = laplace3d_DT_panel_hcubature(panel_src, panel_trg, atol)
        DT_direct = laplace3d_DT_panel(panel_src, panel_trg)
        block = DT_exact - DT_direct

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

# linear operator for the corrected DT kernel
function laplace3d_DT_fmm3d_corrected(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    up_tol::Float64,
    max_order::Int;
    include_edges_src::Bool = false,
    include_edges_trg::Bool = false,
) where {P <: AbstractPanel}
    n_points = num_points(interface)
    D_base = laplace3d_DT_fmm3d(interface, fmm_tol)
    neighbor_list = build_neighbor_list(interface, max_order, up_tol, include_edges_src, include_edges_trg)
    @info "length of neighbor_list: $(length(keys(neighbor_list))) out of $(length(interface.panels)^2)"
    corrections = laplace3d_DT_corrections(interface, neighbor_list)

    f = charges -> (D_base * charges) + (corrections * charges)
    return LinearMap{Float64}(f, n_points, n_points)
end

function laplace3d_DT_fmm3d_corrected_hcubature(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    hcubature_atol::Float64,
    range_factor::Float64;
    include_edges_src::Bool = false,
    include_edges_trg::Bool = false,
) where {P <: AbstractPanel}
    n_points = num_points(interface)
    D_base = laplace3d_DT_fmm3d(interface, fmm_tol)
    neighbor_list = build_neighbor_list(interface, 1, hcubature_atol, include_edges_src, include_edges_trg, distance_only = true, range_factor = range_factor)
    @info "length of neighbor_list: $(length(keys(neighbor_list))) out of $(length(interface.panels)^2)"
    corrections = laplace3d_DT_corrections_hcubature(interface, neighbor_list, hcubature_atol)

    f = charges -> (D_base * charges) + (corrections * charges)
    return LinearMap{Float64}(f, n_points, n_points)
end

function laplace3d_pottrg_near(interface::DielectricInterface{P, T}, target::NTuple{3, T}, sol::AbstractVector{T}, atol::T; range_factor::T = T(5)) where {P <: AbstractPanel, T}
    n_panels = length(interface.panels)
    panel_counts = zeros(Int, n_panels)
    for i in 1:n_panels
        panel_counts[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, panel_counts))
    @assert offsets[end] == length(sol)

    val = zero(T)
    for (i, panel) in enumerate(interface.panels)
        n_quad = panel.n_quad
        a, b, c, d = panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        Lx = norm(b .- a)
        Ly = norm(d .- a)
        l_panel = max(Lx, Ly)
        r_i = range_factor * l_panel / n_quad
        dist = norm(cc .- target)
        idx_start = offsets[i] + 1
        idx_end = offsets[i + 1]
        sol_panel = @view sol[idx_start:idx_end]

        if dist > r_i
            for (j, point) in enumerate(eachpoint(panel))
                val += sol_panel[j] * point.weight * laplace3d_pot(point.point, target)
            end
        else
            ns = panel.gl_xs
            ws = panel.gl_ws
            λ = gl_barycentric_weights(ns, ws)
            bma = b .- a
            dma = d .- a
            scale = Lx * Ly / 4

            function integrand(x)
                u = x[1]
                v = x[2]
                rx = barycentric_row(ns, λ, u)
                ry = barycentric_row(ns, λ, v)
                dens = zero(T)
                idx = 1
                for ii in 1:n_quad
                    for jj in 1:n_quad
                        dens += sol_panel[idx] * rx[ii] * ry[jj]
                        idx += 1
                    end
                end
                p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                return dens * laplace3d_pot(p, target) * scale
            end

            res, _ = hcubature(integrand, T[-1, -1], T[1, 1]; atol = atol)
            val += res
        end
    end

    return val
end
