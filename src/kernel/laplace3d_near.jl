function laplace3d_panel_kernel(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}, mode::Symbol) where T

    np_src = num_points(panel_src)
    np_trg = num_points(panel_trg)

    K = zeros(T, np_trg, np_src)
    if mode === :DT
        for (i, pointi) in enumerate(eachpoint(panel_src))
            for (j, pointj) in enumerate(eachpoint(panel_trg))
                K[j, i] = laplace3d_grad(pointi.point, pointj.point, pointj.normal)
            end
        end
    elseif mode === :D
        for (i, pointi) in enumerate(eachpoint(panel_src))
            for (j, pointj) in enumerate(eachpoint(panel_trg))
                K[j, i] = laplace3d_grad(pointj.point, pointi.point, pointi.normal)
            end
        end
    else
        error("unknown mode for laplace3d_panel_kernel")
    end
    return K
end

function laplace3d_DT_panel(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}) where T
    DT = laplace3d_panel_kernel(panel_src, panel_trg, :DT)
    return DT * diagm(panel_src.weights)
end

function laplace3d_D_panel(panel_src::FlatPanel{T, 3}, panel_trg::FlatPanel{T, 3}) where T
    D = laplace3d_panel_kernel(panel_src, panel_trg, :D)
    return D * diagm(panel_src.weights)
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

    Base.Threads.@threads for ti in 1:np_trg
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
            for ii in 1:n_quad
                for jj in 1:n_quad
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

function laplace3d_pot_panel_hcubature(
    panel_src::FlatPanel{T, 3},
    targets::Matrix{T},
    target_ids::Vector{Int},
    atol::T,
) where T
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
    n_src = n_quad * n_quad
    pot_exact = zeros(T, length(target_ids), n_src)

    Base.Threads.@threads for ti in 1:length(target_ids)
        target_id = target_ids[ti]
        target = (targets[1, target_id], targets[2, target_id], targets[3, target_id])

        function integrand(x)
            u = x[1]
            v = x[2]
            rx = T.(barycentric_row(ns, λ, u))
            ry = T.(barycentric_row(ns, λ, v))
            p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            k = laplace3d_pot(p, target) * scale
            vals = Vector{T}(undef, n_src)
            idx = 1
            for ii in 1:n_quad
                for jj in 1:n_quad
                    vals[idx] = k * rx[ii] * ry[jj]
                    idx += 1
                end
            end
            return vals
        end

        res, _ = hcubature(integrand, T[-1, -1], T[1, 1]; atol = atol)
        @views pot_exact[ti, :] .= res
    end

    return pot_exact
end

function build_target_neighbor_list(
    interface::DielectricInterface{P, T},
    targets::Matrix{T},
    include_edges_src::Bool;
    range_factor::T = T(5),
) where {P <: AbstractPanel, T}
    @assert size(targets, 1) == 3
    neighbor_list = Dict{Int, Vector{Int}}()
    tree = KDTree(targets)

    for (i, panel) in enumerate(interface.panels)
        (!include_edges_src && panel.is_edge) && continue

        c_panel = (panel.corners[1] .+ panel.corners[2] .+ panel.corners[3] .+ panel.corners[4]) ./ 4
        l_panel = max(norm(panel.corners[1] .- panel.corners[2]), norm(panel.corners[2] .- panel.corners[3]))
        r_i = range_factor * l_panel / panel.n_quad

        nearby = inrange(tree, collect(c_panel), r_i)
        isempty(nearby) && continue
        neighbor_list[i] = nearby
    end

    return neighbor_list
end

function _panel_center(panel::FlatPanel{T, 3}) where T
    return (panel.corners[1] .+ panel.corners[2] .+ panel.corners[3] .+ panel.corners[4]) ./ 4
end

function _panel_max_length(panel::FlatPanel{T, 3}) where T
    return max(norm(panel.corners[1] .- panel.corners[2]), norm(panel.corners[2] .- panel.corners[3]))
end

function _split_panel4(panel::FlatPanel{T, 3}) where T
    a, b, c, d = panel.corners
    ab = (a .+ b) ./ 2
    bc = (b .+ c) ./ 2
    cd = (c .+ d) ./ 2
    da = (d .+ a) ./ 2
    cc = (a .+ b .+ c .+ d) ./ 4
    ns = panel.gl_xs
    ws = panel.gl_ws
    normal = panel.normal
    is_edge = panel.is_edge
    return FlatPanel{T, 3}[
        rect_panel3d_discretize(a, ab, cc, da, ns, ws, normal; is_edge = is_edge),
        rect_panel3d_discretize(ab, b, bc, cc, ns, ws, normal; is_edge = is_edge),
        rect_panel3d_discretize(cc, bc, c, cd, ns, ws, normal; is_edge = is_edge),
        rect_panel3d_discretize(da, cc, cd, d, ns, ws, normal; is_edge = is_edge),
    ]
end

function _refine_interface_for_targets(
    interface::DielectricInterface{FlatPanel{T, 3}, T},
    targets::Matrix{T},
    panel_size_limit::T;
    range_factor::T = T(5),
) where T
    @assert size(targets, 1) == 3

    if isempty(interface.panels)
        parent_ids = collect(1:length(interface.panels))
        from_split = fill(false, length(interface.panels))
        return interface, parent_ids, from_split
    end

    tree = KDTree(targets)
    stack = Tuple{Int, FlatPanel{T, 3}, Bool}[]
    for i in eachindex(interface.panels)
        push!(stack, (i, interface.panels[i], false))
    end

    refined_panels = FlatPanel{T, 3}[]
    parent_ids = Int[]
    from_split = Bool[]
    while !isempty(stack)
        parent_idx, panel, is_split = pop!(stack)
        l_panel = _panel_max_length(panel)
        c_panel = _panel_center(panel)
        r_i = range_factor * l_panel / panel.n_quad
        near_targets = inrange(tree, collect(c_panel), r_i)

        if !isempty(near_targets) && l_panel > panel_size_limit
            for child in _split_panel4(panel)
                push!(stack, (parent_idx, child, true))
            end
        else
            push!(refined_panels, panel)
            push!(parent_ids, parent_idx)
            push!(from_split, is_split)
        end
    end

    eps_in = Vector{T}(undef, length(refined_panels))
    eps_out = Vector{T}(undef, length(refined_panels))
    for i in eachindex(refined_panels)
        pid = parent_ids[i]
        eps_in[i] = interface.eps_in[pid]
        eps_out[i] = interface.eps_out[pid]
    end

    return DielectricInterface(refined_panels, eps_in, eps_out), parent_ids, from_split
end

function _refined_interface_prolongation(
    coarse_interface::DielectricInterface{FlatPanel{T, 3}, T},
    refined_interface::DielectricInterface{FlatPanel{T, 3}, T},
    parent_ids::Vector{Int},
    from_split::Vector{Bool},
) where T
    length(parent_ids) == length(refined_interface.panels) || throw(ArgumentError("parent_ids length mismatch"))
    length(from_split) == length(refined_interface.panels) || throw(ArgumentError("from_split length mismatch"))

    n_coarse = num_points(coarse_interface)
    n_refined = num_points(refined_interface)

    coarse_counts = [length(panel.points) for panel in coarse_interface.panels]
    coarse_offsets = cumsum(vcat(0, coarse_counts))

    rows = Int[]
    cols = Int[]
    vals = T[]
    row_id = 0

    for (ref_idx, panel_ref) in enumerate(refined_interface.panels)
        parent_panel = coarse_interface.panels[parent_ids[ref_idx]]
        col0 = coarse_offsets[parent_ids[ref_idx]]

        if !from_split[ref_idx]
            for k in 1:length(panel_ref.points)
                row_id += 1
                push!(rows, row_id)
                push!(cols, col0 + k)
                push!(vals, one(T))
            end
            continue
        end

        ns = parent_panel.gl_xs
        ws = parent_panel.gl_ws
        λ = gl_barycentric_weights(ns, ws)
        n_quad = parent_panel.n_quad
        a, b, c, d = parent_panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a
        dma = d .- a
        bb = dot(bma, bma)
        bd = dot(bma, dma)
        dd = dot(dma, dma)
        det = bb * dd - bd * bd
        inv11 = dd / det
        inv12 = -bd / det
        inv22 = bb / det

        for p in panel_ref.points
            row_id += 1
            x = p .- cc
            rhs1 = dot(bma, x)
            rhs2 = dot(dma, x)
            s = inv11 * rhs1 + inv12 * rhs2
            t = inv12 * rhs1 + inv22 * rhs2
            u = 2 * s
            v = 2 * t
            rx = T.(barycentric_row(ns, λ, u))
            ry = T.(barycentric_row(ns, λ, v))
            for ii in 1:n_quad
                for jj in 1:n_quad
                    col_id = col0 + (ii - 1) * n_quad + jj
                    push!(rows, row_id)
                    push!(cols, col_id)
                    push!(vals, rx[ii] * ry[jj])
                end
            end
        end
    end

    return sparse(rows, cols, vals, n_refined, n_coarse)
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

function laplace3d_pottrg_corrections_hcubature(
    interface::DielectricInterface{P, T},
    targets::Matrix{T},
    target_neighbor_list::Dict{Int, Vector{Int}},
    atol::T,
) where {P <: AbstractPanel, T}
    cnt = zeros(Int, length(interface.panels))
    for i in 1:length(interface.panels)
        cnt[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]
    n_targets = size(targets, 2)

    rows = Int[]
    cols = Int[]
    vals = T[]

    for (i, target_ids) in target_neighbor_list
        panel_src = interface.panels[i]
        col_range = (offsets[i] + 1):offsets[i + 1]

        pot_exact = laplace3d_pot_panel_hcubature(panel_src, targets, target_ids, atol)

        for (t_local, t_global) in enumerate(target_ids)
            target = (targets[1, t_global], targets[2, t_global], targets[3, t_global])
            for (c_local, c_global) in enumerate(col_range)
                direct = panel_src.weights[c_local] * laplace3d_pot(panel_src.points[c_local], target)
                v = pot_exact[t_local, c_local] - direct
                iszero(v) && continue
                push!(rows, t_global)
                push!(cols, c_global)
                push!(vals, v)
            end
        end
    end

    return sparse(rows, cols, vals, n_targets, total_n)
end

# direct evaluation of correction action (DT_exact - DT_direct) * sigma
function laplace3d_DT_corrections_hcubature_apply(
    interface::DielectricInterface{P, T},
    neighbor_list::Dict{Tuple{Int, Int}, Int},
    atol::T,
    sigma::Function,
) where {P <: AbstractPanel, T}
    cnt = zeros(Int, length(interface.panels))
    for i in 1:length(interface.panels)
        cnt[i] = length(interface.panels[i].points)
    end
    offsets = cumsum(vcat(0, cnt))
    total_n = offsets[end]

    out = zeros(T, total_n)

    for ((i, j), _) in neighbor_list
        panel_src = interface.panels[i]
        panel_trg = interface.panels[j]

        # precompute sigma at source quadrature nodes for the direct term
        sigma_src = Vector{T}(undef, length(panel_src.points))
        for k in 1:length(panel_src.points)
            sigma_src[k] = T(sigma(panel_src.points[k]))
        end

        a, b, c, d = panel_src.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a
        dma = d .- a
        Lx = norm(b .- a)
        Ly = norm(d .- a)
        scale = Lx * Ly / 4

        weights_src = panel_src.weights
        points_src = panel_src.points
        points_trg = panel_trg.points
        normal_trg = panel_trg.normal

        row_range = (offsets[j] + 1):offsets[j + 1]

        for (t_local, t_global) in enumerate(row_range)
            point_trg = points_trg[t_local]

            function integrand(x)
                u = x[1]
                v = x[2]
                p = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
                k = laplace3d_grad(p, point_trg, normal_trg) * scale
                return T(sigma(p)) * k
            end

            exact, _ = hcubature(integrand, T[-1, -1], T[1, 1]; atol = atol)

            direct = zero(T)
            @inbounds for s in 1:length(points_src)
                direct += sigma_src[s] * weights_src[s] * laplace3d_grad(points_src[s], point_trg, normal_trg)
            end

            out[t_global] += exact - direct
        end
    end

    return out
end

# linear operator for the corrected DT kernel
function laplace3d_DT_fmm3d_corrected(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    up_tol::Float64,
    max_order::Int;
    include_edges_src::Bool = false,
    include_edges_trg::Bool = false,
    range_factor::Float64 = 5.0,
) where {P <: AbstractPanel}
    n_points = num_points(interface)
    D_base = laplace3d_DT_fmm3d(interface, fmm_tol)
    neighbor_list = build_neighbor_list(interface, max_order, up_tol, include_edges_src, include_edges_trg, range_factor = range_factor)
    @info "length of neighbor_list: $(length(keys(neighbor_list))) out of $(length(interface.panels)^2)"
    corrections = laplace3d_DT_corrections(interface, neighbor_list)

    f = charges -> (D_base * charges) + (corrections * charges)
    return LinearMap{Float64}(f, n_points, n_points)
end

function laplace3d_D_fmm3d_corrected(
    interface::DielectricInterface{P, Float64},
    fmm_tol::Float64,
    up_tol::Float64,
    max_order::Int;
    include_edges_src::Bool = false,
    include_edges_trg::Bool = false,
) where {P <: AbstractPanel}
    n_points = num_points(interface)
    D_base = laplace3d_D_fmm3d(interface, fmm_tol)
    neighbor_list = build_neighbor_list(interface, max_order, up_tol, include_edges_src, include_edges_trg)
    @info "length of neighbor_list: $(length(keys(neighbor_list))) out of $(length(interface.panels)^2)"
    corrections = laplace3d_D_corrections(interface, neighbor_list)

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

function laplace3d_pottrg_fmm3d_corrected_hcubature(
    interface::DielectricInterface{P, Float64},
    targets::Matrix{Float64},
    fmm_tol::Float64,
    hcubature_atol::Float64,
    range_factor::Float64;
    include_edges_src::Bool = false,
) where {P <: AbstractPanel}
    @assert size(targets, 1) == 3

    n_points = num_points(interface)
    refined_interface = interface
    prolongation = sparse(1:n_points, 1:n_points, ones(Float64, n_points), n_points, n_points)
    if P <: FlatPanel{Float64, 3}
        panel_size_limit = minimum(_panel_max_length(panel) for panel in interface.panels)
        refined_interface, parent_ids, from_split = _refine_interface_for_targets(
            interface,
            targets,
            panel_size_limit;
            range_factor = range_factor,
        )
        prolongation = _refined_interface_prolongation(interface, refined_interface, parent_ids, from_split)
    end

    pot_base = laplace3d_pottrg_fmm3d(refined_interface, targets, fmm_tol)
    target_neighbor_list = build_target_neighbor_list(refined_interface, targets, include_edges_src; range_factor = range_factor)
    
    @info "num of sources: $(num_points(interface)) → $(num_points(refined_interface))"
    if !isempty(target_neighbor_list)
        @info "num of hcub calculations: $(sum(length(v) for v in values(target_neighbor_list)))"
    end

    corrections = laplace3d_pottrg_corrections_hcubature(refined_interface, targets, target_neighbor_list, hcubature_atol)

    f = charges -> begin
        charges_refined = prolongation * charges
        return (pot_base * charges_refined) + (corrections * charges_refined)
    end
    return LinearMap{Float64}(f, size(targets, 2), n_points)
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
