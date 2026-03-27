# internal panel type for 2d straight line panel generation
struct TempPanel2D{T}
    a::NTuple{2, T}
    b::NTuple{2, T}

    is_a_corner::Bool
    is_b_corner::Bool

    normal::NTuple{2, T}
end

# 1d line panel with Guass-Legendre quadrature
function line_panel2d_discretize(a::NTuple{2, T}, b::NTuple{2, T}, ns::Vector{T}, ws::Vector{T}, normal::NTuple{2, T}; is_edge::Bool = true) where T

    points = [(b .+ a) ./ 2 .+ ns[i] .* (b .- a) ./ 2 for i in 1:length(ns)]
    L = norm(b .- a)
    weights = ws .* L ./ 2

    @assert norm(normal) ≈ 1 "Normal is not a unit vector"
    @assert dot(normal, b .- a) < 1e-10 "Normal is not perpendicular to the line segment"

    return FlatPanel(normal, [a, b], is_edge, length(ns), ns, ws, points, weights)
end

# divide the panel into two smaller panels
function divide_temp_panel2d(tpl::TempPanel2D{T}, n_divide::Int) where T
    @assert n_divide >= 2 "n_divide must be greater than or equal to 2"

    panels = Vector{TempPanel2D{T}}(undef, n_divide)

    for i in 1:n_divide
        p_start = tpl.a .+ (tpl.b .- tpl.a) .* (i - 1) ./ n_divide
        p_end = tpl.a .+ (tpl.b .- tpl.a) .* i ./ n_divide
        is_corner_left = (i == 1) ? tpl.is_a_corner : false
        is_corner_right = (i == n_divide) ? tpl.is_b_corner : false
        panels[i] = TempPanel2D(p_start, p_end, is_corner_left, is_corner_right, tpl.normal)
    end

    return panels
end

# assume that the input is the starting and ending points of a line, which is to be divided into a vector of panels
# l_panel is the maximum length of a none-corner panel, l_corner is the maximum length of a corner panel
function straight_line_adaptive_panels(sp::NTuple{2, T}, ep::NTuple{2, T}, ns::Vector{T}, ws::Vector{T}, normal::NTuple{2, T}, l_panel::T, l_corner::T) where T

    l_line = norm(ep .- sp)
    n_divide_rough = ceil(Int, l_line / l_panel)

    # this gives a rough division
    rough_panels = divide_temp_panel2d(TempPanel2D(sp, ep, true, true, normal), n_divide_rough)
    fine_panels = Vector{TempPanel2D{T}}()

    while !isempty(rough_panels)
        tpl = popfirst!(rough_panels)
        # check if the panel is a corner panel
        if tpl.is_a_corner || tpl.is_b_corner
            # if the panel is a corner panel, and the length is less than or equal to l_corner, then it is a fine panel
            panel_length = norm(tpl.a .- tpl.b)
            if panel_length <= l_corner
                push!(fine_panels, tpl)
            else # refine the panel by two if the length is greater than l_corner
                refined_panels = divide_temp_panel2d(tpl, 2)
                append!(rough_panels, refined_panels)
            end
        else
            # if the panel is a none-corner panel, then it is a fine panel (we already have a rough division)
            push!(fine_panels, tpl)
        end
    end

    # discretize the fine panels
    panels = Vector{FlatPanel{T, 2}}()
    for tpl in fine_panels
        push!(panels, line_panel2d_discretize(tpl.a, tpl.b, ns, ws, tpl.normal))
    end

    return panels
end

function single_dielectric_box2d(Lx::T, Ly::T, n_quad::Int, l_panel::T, l_corner::T, eps_in::T, eps_out::T, ::Type{T} = Float64) where T
    ns, ws = gausslegendre(n_quad)
    hx = Lx / 2
    hy = Ly / 2
    t0 = zero(T)
    panels = Vector{FlatPanel{T, 2}}()
    for (sp, ep, normal) in zip([(-hx, hy), (hx, hy), (hx, -hy), (-hx, -hy)], [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)], [(t0, one(T)), (one(T), t0), (t0, -one(T)), (-one(T), t0)])
        append!(panels, straight_line_adaptive_panels(sp, ep, ns, ws, normal, l_panel, l_corner))
    end

    return DielectricInterface(panels, fill(eps_in, length(panels)), fill(eps_out, length(panels)))

end

square = (x, y) -> [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
rect = (x, y, w, h) -> [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

# vertices in the same box are arranged counter-clockwise, for example: [(0,0), (1,0), (1,1), (0,1)]
# id of the box is the index of the box in the vector, id of vacuum is 0
# norm vector points from box with higher id to box with lower id
function multi_dielectric_box2d(n_quad::Int, l_panel::T, l_corner::T, vec_boxes::Vector{Vector{NTuple{2, T}}}, epses::Vector{T}) where T
    ns, ws = gausslegendre(n_quad)
    ns = T.(ns)
    ws = T.(ws)

    edges = build_edges(vec_boxes)
    interfaces_dict = Dict{Tuple{Int, Int}, Vector{FlatPanel{T, 2}}}()
    for (p1, p2, id1, id2, nvec) in edges
        panels = straight_line_adaptive_panels(p1, p2, ns, ws, nvec, l_panel, l_corner)
        if haskey(interfaces_dict, (id1, id2))
            append!(interfaces_dict[(id1, id2)], panels)
        else
            interfaces_dict[(id1, id2)] = panels
        end
    end

    panels_vec = Vector{FlatPanel{T, 2}}()
    eps_in_vec = Vector{T}()
    eps_out_vec = Vector{T}()
    for (ids, panels) in interfaces_dict
        id1, id2 = ids

        eps_in = iszero(id1) ? one(T) : epses[id1]
        eps_out = iszero(id2) ? one(T) : epses[id2]

        append!(panels_vec, panels)
        append!(eps_in_vec, fill(eps_in, length(panels)))
        append!(eps_out_vec, fill(eps_out, length(panels)))
    end

    return DielectricInterface(panels_vec, eps_in_vec, eps_out_vec)
end

function edge_normal(p1, p2)
    v = [p2[1] - p1[1], p2[2] - p1[2]]
    n = [v[2], -v[1]]
    n ./= norm(n)
    return tuple(n...)
end

function normalize_edge(p1, p2)
    return (p1 < p2) ? (p1, p2) : (p2, p1)
end

function overlap_segment(p1, p2, p3, p4; tol = 1e-10)
    v1 = [p2[1] - p1[1], p2[2] - p1[2]]
    cross1 = abs(det([v1 [p3[1] - p1[1]; p3[2] - p1[2]]]))
    cross2 = abs(det([v1 [p4[1] - p1[1]; p4[2] - p1[2]]]))
    if cross1 > tol || cross2 > tol
        return false, nothing
    end

    if abs(v1[1]) >= abs(v1[2])
        pts = sort!([p1, p2, p3, p4], by = x -> x[1])
        lo, hi = pts[2], pts[3]
    else
        pts = sort!([p1, p2, p3, p4], by = x -> x[2])
        lo, hi = pts[2], pts[3]
    end

    if (hi[1] - lo[1])^2 + (hi[2] - lo[2])^2 > tol^2
        return true, (lo, hi)
    end
    return false, nothing
end

function intersect_edges(rect1::Vector{NTuple{2, T}}, rect2::Vector{NTuple{2, T}}) where T
    for (v11, v12) in zip(rect1, circshift(rect1, -1))
        for (v21, v22) in zip(rect2, circshift(rect2, -1))
            overlap, lohi = overlap_segment(v11, v12, v21, v22)
            if overlap
                nvec = edge_normal(v21, v22)
                return true, (lohi[1], lohi[2], nvec)
            end
        end
    end
    return false, nothing
end

function split_edge_by_overlaps(p1::NTuple{2, T}, p2::NTuple{2, T}, overlaps::Vector{Tuple{NTuple{2, T}, NTuple{2, T}}}; tol = 1e-10) where T
    if abs(p2[1] - p1[1]) >= abs(p2[2] - p1[2])
        smin, smax = min(p1[1], p2[1]), max(p1[1], p2[1])
        proj = x -> x[1]
    else
        smin, smax = min(p1[2], p2[2]), max(p1[2], p2[2])
        proj = x -> x[2]
    end

    intervals = Tuple{T, T}[]
    for (q1, q2) in overlaps
        lo, hi = sort([proj(q1), proj(q2)])
        push!(intervals, (lo, hi))
    end

    sort!(intervals, by = i -> i[1])
    merged = Tuple{T, T}[]
    for iv in intervals
        if isempty(merged) || iv[1] > merged[end][2] + tol
            push!(merged, iv)
        else
            merged[end] = (merged[end][1], max(merged[end][2], iv[2]))
        end
    end

    segments = Tuple{T, T}[]
    cursor = smin
    for (lo, hi) in merged
        if lo - cursor > tol
            push!(segments, (cursor, lo))
        end
        cursor = hi
    end
    if smax - cursor > tol
        push!(segments, (cursor, smax))
    end

    res = Tuple{NTuple{2, T}, NTuple{2, T}}[]
    for (a, b) in segments
        if abs(p2[1] - p1[1]) >= abs(p2[2] - p1[2])
            yA = p1[2] + (a - p1[1]) / (p2[1] - p1[1]) * (p2[2] - p1[2])
            yB = p1[2] + (b - p1[1]) / (p2[1] - p1[1]) * (p2[2] - p1[2])
            push!(res, ((a, yA), (b, yB)))
        else
            xA = p1[1] + (a - p1[2]) / (p2[2] - p1[2]) * (p2[1] - p1[1])
            xB = p1[1] + (b - p1[2]) / (p2[2] - p1[2]) * (p2[1] - p1[1])
            push!(res, ((xA, a), (xB, b)))
        end
    end
    return res
end

function build_edges(rects::Vector{Vector{NTuple{2, T}}}) where T
    edges = Tuple{NTuple{2, T}, NTuple{2, T}, Int, Int, NTuple{2, T}}[]
    shared_edges = Dict{Tuple{NTuple{2, T}, NTuple{2, T}}, Vector{Int}}()

    for i in 1:length(rects) - 1
        for j in i + 1:length(rects)
            overlap, lohinvec = intersect_edges(rects[i], rects[j])
            if overlap
                p1, p2 = lohinvec[1], lohinvec[2]
                nvec = lohinvec[3]
                key = normalize_edge(p1, p2)
                shared_edges[key] = union(get(shared_edges, key, Int[]), [i, j])
                push!(edges, (p1, p2, j, i, nvec))
            end
        end
    end

    for (rid, rect) in enumerate(rects)
        for (p1, p2) in zip(rect, circshift(rect, 1))
            key = normalize_edge(p1, p2)
            nvec = edge_normal(p1, p2)

            overlaps = Tuple{NTuple{2, T}, NTuple{2, T}}[]
            for (k, regs) in shared_edges
                if rid in regs
                    overlap, seg = overlap_segment(p1, p2, k[1], k[2])
                    if overlap
                        push!(overlaps, seg)
                    end
                end
            end

            remain = split_edge_by_overlaps(p1, p2, overlaps)
            for (q1, q2) in remain
                push!(edges, (q1, q2, rid, 0, (-nvec[1], -nvec[2])))
            end
        end
    end
    return edges
end
